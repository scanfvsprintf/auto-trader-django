# ==============================================================================
# 文件 2/4: selection_manager/management/commands/train_stock_model.py
# 描述: 训练个股评分的LightGBM回归模型。
# ==============================================================================
import logging
import pickle
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from django.core.management.base import BaseCommand
from django.conf import settings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import optuna

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '训练个股评分的LightGBM回归模型。'

    # --- 路径配置 ---
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    DATASET_FILE = MODELS_DIR / 'stock_features_dataset.pkl'
    MODEL_FILE = MODELS_DIR / 'stock_lgbm_model.joblib'
    MODEL_CONFIG_FILE = MODELS_DIR / 'stock_model_config.json'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("===== 开始训练个股评分模型 (LightGBM Regressor) ====="))

        # 1. 加载数据集
        self.stdout.write(f"步骤 1/4: 从 {self.DATASET_FILE} 加载数据集...")
        try:
            with open(self.DATASET_FILE, 'rb') as f:
                dataset = pickle.load(f)
            X, y, feature_names = dataset['X'], dataset['y'], dataset['feature_names']
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"错误: 数据集文件 {self.DATASET_FILE} 不存在。请先运行 'prepare_stock_features' 命令。"))
            return

        # 2. 定义模型和交叉验证
        self.stdout.write("步骤 2/4: 设置回归模型参数和时间序列交叉验证...")
        
        # LightGBM 回归模型参数 (与M值模型风格一致)
        lgbm_params = {
            'objective': 'regression_l1', # 使用L1损失 (MAE)，对异常值更鲁棒
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'n_estimators': 70000, # 减少迭代次数，因为数据集更大
            'learning_rate': 0.001,
            'n_jobs': -1,
            'seed': 42,
            'verbose': -1,
            'num_leaves': 63, # 允许更复杂的树
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'num_leaves': 978, 'learning_rate': 0.01, 'subsample': 0.741886904800338, 'colsample_bytree': 0.6802622809997809, 'reg_alpha': 1.306972432246392, 'reg_lambda': 0.06447061402735922, 'min_child_samples': 100
        }

        # 3. 使用 Optuna 进行超参数优化和验证
        self.stdout.write("步骤 3/4: 使用 Optuna 进行超参数优化...")
        
        # 由于数据集可能非常大，我们只取最后一部分数据进行快速的超参数搜索
        # 例如，只用最后20%的数据进行调优
        sample_frac_for_tuning = 0.2
        tuning_data_size = int(len(X) * sample_frac_for_tuning)
        X_tuning = X.iloc[-tuning_data_size:]
        y_tuning = y.iloc[-tuning_data_size:]
        self.stdout.write(f"为加速调优，仅使用后 {tuning_data_size} 条数据进行参数搜索。")

        # def objective(trial):
        #     params_to_tune = {
        #         'num_leaves': trial.suggest_int('num_leaves', 100, 1000),
        #         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        #         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        #         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        #         'reg_alpha': trial.suggest_float('reg_alpha', 1e-1, 1000.0, log=True),
        #         'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 1000.0, log=True),
        #         'min_child_samples': trial.suggest_int('min_child_samples', 100, 500)
        #     }
        #     current_params = lgbm_params.copy()
        #     current_params.update(params_to_tune)
            
        #     tscv = TimeSeriesSplit(n_splits=5)
        #     rmses = []
        #     for train_index, val_index in tscv.split(X_tuning):
        #         X_train_fold, X_val_fold = X_tuning.iloc[train_index], X_tuning.iloc[val_index]
        #         y_train_fold, y_val_fold = y_tuning.iloc[train_index], y_tuning.iloc[val_index]
                
        #         model = lgb.LGBMRegressor(**current_params)
        #         model.fit(X_train_fold, y_train_fold,
        #                   eval_set=[(X_val_fold, y_val_fold)],
        #                   callbacks=[lgb.early_stopping(50, verbose=False)])
                
        #         val_preds = model.predict(X_val_fold)
        #         rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
        #         rmses.append(rmse)
        #     return np.mean(rmses)

        # self.stdout.write("开始参数搜索 (n_trials=30)...")
        # study = optuna.create_study(direction='minimize')
        # study.optimize(objective, n_trials=10) # 减少尝试次数以适应大数据集
        
        best_params = lgbm_params.copy()
        # best_params.update(study.best_params)
        # self.stdout.write(self.style.SUCCESS(f"\n搜索完成！最佳验证集 RMSE: {study.best_value:.4f}"))
        # self.stdout.write(self.style.SUCCESS(f"找到的最佳参数: {study.best_params}"))

        # 步骤 4/4: 使用最佳参数在全部数据上训练最终模型并保存
        self.stdout.write("\n步骤 4/4: 使用最佳参数在全部数据上训练最终模型并保存...")
        
        # 使用时间序列划分法，将最后20%作为验证集来展示最终性能
        split_point = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]

        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric='rmse',
                        callbacks=[lgb.early_stopping(100, verbose=True)])
        
        # 在验证集上评估最终模型
        val_preds = final_model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_r2 = r2_score(y_val, val_preds)
        self.stdout.write(self.style.SUCCESS(f"\n--- 最终模型在验证集上的性能 ---"))
        self.stdout.write(f"RMSE: {val_rmse:.4f}")
        self.stdout.write(f"R^2 Score: {val_r2:.4f}")

        # 保存模型
        joblib.dump(final_model, self.MODEL_FILE)
        self.stdout.write(self.style.SUCCESS(f"最终模型已保存至: {self.MODEL_FILE}"))
        
        # 显示并保存特征重要性
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        self.stdout.write("\n--- 特征重要性 ---")
        self.stdout.write(str(feature_importance_df.head(20)))
        
        # 更新模型配置文件
        try:
            with open(self.MODEL_CONFIG_FILE, 'r') as f:
                model_config = json.load(f)
            
            model_config['feature_importance'] = feature_importance_df.to_dict('records')
            #model_config['best_params'] = {k: v for k, v in study.best_params.items()}
            
            with open(self.MODEL_CONFIG_FILE, 'w') as f:
                json.dump(model_config, f, indent=4)
            self.stdout.write(self.style.SUCCESS(f"特征重要性和最佳参数已更新至: {self.MODEL_CONFIG_FILE}"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"更新模型配置文件时出错: {e}"))
            
        self.stdout.write(self.style.SUCCESS("===== 个股评分模型训练流程结束！ ====="))
