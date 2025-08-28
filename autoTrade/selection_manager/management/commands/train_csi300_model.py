# selection_manager/management/commands/train_csi300_model.py

import logging
import pickle
import json # 新增
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '[Final] 训练单个CNN模型，并寻找最佳决策阈值。'
    
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    FEATURES_FILE = MODELS_DIR / 'csi300_features_gen2.pkl'
    MODEL_FILE = MODELS_DIR / 'csi300_cnn_final_model.h5'
    CONFIG_FILE = MODELS_DIR / 'csi300_model_config.json' # [新] 保存配置，包括阈值

    N_FEATURES = 10
    LOOKBACK_WINDOW = 60
    N_CLASSES = 2

    def build_model(self):
        """模型结构可以保持不变，或根据需要微调"""
        model = Sequential([
            Input(shape=(self.LOOKBACK_WINDOW, self.N_FEATURES)),
            BatchNormalization(),
            Conv1D(filters=32, kernel_size=5, activation='relu', padding='causal'),
            BatchNormalization(),
            Dropout(0.3),
            Conv1D(filters=16, kernel_size=5, activation='relu', padding='causal'),
            GlobalMaxPooling1D(),
            Dropout(0.5),
            Dense(self.N_CLASSES, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        return model
    
    def find_best_threshold(self, y_true, y_pred_proba):
        """遍历寻找最佳阈值"""
        best_threshold = 0.5
        best_f1_macro = 0.0
        
        # y_pred_proba[:, 0] 是上涨的概率
        up_probabilities = y_pred_proba[:, 0]

        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred_class = (up_probabilities >= threshold).astype(int)
            # 注意：因为我们的模型输出是[P_up, P_down]，所以P_up >= threshold时，预测为类别0（上涨）
            # Scikit-learn 的 f1_score 期望的是 0 和 1
            # 当 P_up >= threshold, 预测为0 (上涨), 否则为1 (下跌)
            # 这与我们的标签定义一致
            current_f1 = f1_score(y_true, (up_probabilities < threshold).astype(int), average='macro')
            
            if current_f1 > best_f1_macro:
                best_f1_macro = current_f1
                best_threshold = threshold
        
        return best_threshold, best_f1_macro

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("===== [Final] 开始训练并优化模型 ====="))

        with open(self.FEATURES_FILE, 'rb') as f:
            feature_data = pickle.load(f)
        X, y = feature_data['X'], feature_data['y']

        tss = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tss.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        y_train_cat = to_categorical(y_train, num_classes=self.N_CLASSES)
        y_test_cat = to_categorical(y_test, num_classes=self.N_CLASSES)

        model = self.build_model()
        model.summary()
        
        early_stopping = EarlyStopping(monitor='val_auc', patience=20, mode='max', restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(self.MODEL_FILE, monitor='val_auc', save_best_only=True, mode='max')
        
        model.fit(
            X_train, y_train_cat,
            epochs=150, batch_size=64,
            validation_data=(X_test, y_test_cat),
            callbacks=[early_stopping, model_checkpoint],
            verbose=2
        )
        
        self.stdout.write(self.style.SUCCESS("\n--- 评估报告 (使用默认0.5阈值) ---"))
        model.load_weights(self.MODEL_FILE)
        y_pred_proba = model.predict(X_test)
        y_pred_class_default = np.argmax(y_pred_proba, axis=1) # 默认阈值等价于argmax
        self.stdout.write(classification_report(y_test, y_pred_class_default, target_names=['上涨', '下跌']))

        self.stdout.write(self.style.SUCCESS("\n--- 寻找并评估最佳阈值 ---"))
        best_threshold, best_f1 = self.find_best_threshold(y_test, y_pred_proba)
        self.stdout.write(f"在验证集上找到的最佳阈值为: {best_threshold:.4f} (F1-macro: {best_f1:.4f})")
        
        y_pred_class_best = (y_pred_proba[:, 0] < best_threshold).astype(int)
        self.stdout.write("\n--- 评估报告 (使用最佳阈值) ---")
        self.stdout.write(classification_report(y_test, y_pred_class_best, target_names=['上涨', '下跌']))

        self.stdout.write("\n--- 保存模型和配置 ---")
        model_config = {
            'best_threshold': best_threshold
        }
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(model_config, f)
            
        self.stdout.write(self.style.SUCCESS(f"最佳模型已保存至: {self.MODEL_FILE}"))
        self.stdout.write(self.style.SUCCESS(f"模型配置 (含阈值) 已保存至: {self.CONFIG_FILE}"))
        self.stdout.write(self.style.SUCCESS("===== [Final] 训练流程结束！ ====="))
