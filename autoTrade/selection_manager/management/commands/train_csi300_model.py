# selection_manager/management/commands/train_csi300_model.py

import logging
import pickle
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from django.core.management.base import BaseCommand
from django.conf import settings
from pathlib import Path

logger = logging.getLogger(__name__)

# [修复] 类定义必须存在且正确
class Command(BaseCommand):
    help = '读取特征文件，训练沪深300预测模型并保存。'
    
    # 路径应与prepare命令中的一致
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    FEATURES_FILE = MODELS_DIR / 'csi300_features.pkl'
    MODEL_FILE = MODELS_DIR / 'csi300_lgbm_predictor.joblib'
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("===== 开始训练沪深300指数预测模型 ====="))

        # 1. 加载特征数据
        self.stdout.write(f"步骤 1/4: 从 {self.FEATURES_FILE} 加载特征数据...")
        if not self.FEATURES_FILE.exists():
            self.stderr.write(self.style.ERROR(
                f"特征文件未找到！请先运行 'python manage.py prepare_csi300_features' 命令。"
            ))
            return
            
        try:
            with open(self.FEATURES_FILE, 'rb') as f:
                feature_data = pickle.load(f)
            X = feature_data['X']
            y = feature_data['y']
            self.stdout.write(f"成功加载 {X.shape[0]} 条样本，每个样本特征维度为 {X.shape[1]}。")
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"加载特征文件失败: {e}"))
            return

        # 2. 划分数据集 (时间序列数据，禁止打乱)
        self.stdout.write("步骤 2/4: 按时间顺序划分训练集和测试集 (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        self.stdout.write(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条。")

        # 3. 训练模型
        self.stdout.write("步骤 3/4: 使用 LightGBM 模型进行训练...")
        model = lgb.LGBMRegressor(
            objective='regression_l1',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            colsample_bytree=0.8,
            subsample=0.8,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(100, verbose=False)] # [优化] 将 verbose 设置为 False，让日志更干净
        )
        
        self.stdout.write("模型训练完成。")
        
        # 4. 评估并保存模型
        self.stdout.write("步骤 4/4: 在测试集上评估并保存模型...")
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        directional_accuracy = np.mean(np.sign(y_pred) == np.sign(y_test))
        
        self.stdout.write(self.style.SUCCESS(f"模型评估完成："))
        self.stdout.write(f"  - 均方误差 (MSE): {mse:.8f}")
        self.stdout.write(f"  - 预测方向准确率: {directional_accuracy:.2%}")

        try:
            joblib.dump(model, self.MODEL_FILE)
            self.stdout.write(self.style.SUCCESS(f"模型已成功保存至: {self.MODEL_FILE}"))
            self.stdout.write(self.style.SUCCESS("===== 训练流程结束！ ====="))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"保存模型文件失败: {e}"))
