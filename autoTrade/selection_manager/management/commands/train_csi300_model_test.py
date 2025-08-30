# selection_manager/management/commands/train_m_value_model.py

import logging
import pickle
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from django.core.management.base import BaseCommand
from django.conf import settings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import optuna

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '[M-Value Refactor] 训练LightGBM模型以识别市场状态（牛/熊/震荡）。'

    # --- 路径配置 ---
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    DATASET_FILE = MODELS_DIR / 'm_value_dataset.pkl'
    MODEL_FILE = MODELS_DIR / 'm_value_lgbm_model.joblib'
    MODEL_CONFIG_FILE = MODELS_DIR / 'm_value_model_config.json'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("===== [M-Value Refactor] 开始训练LightGBM模型 ====="))

        # 1. 加载数据集
        self.stdout.write(f"步骤 1/4: 从 {self.DATASET_FILE} 加载数据集...")
        try:
            with open(self.DATASET_FILE, 'rb') as f:
                dataset = pickle.load(f)
            X, y, feature_names = dataset['X'], dataset['y'], dataset['feature_names']
            label_map = dataset['label_map']
            target_names = [label_map[i] for i in sorted(label_map.keys())]
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"错误: 数据集文件 {self.DATASET_FILE} 不存在。请先运行 'prepare_m_value_features' 命令。"))
            return

        # 2. 定义模型和交叉验证
        self.stdout.write("步骤 2/4: 设置模型参数和时间序列交叉验证...")
        
        # LightGBM 参数 (可根据需要进行调优)
        lgbm_params = {
            # --- 核心参数 (定义模型的基本任务和类型) ---
            'objective': 'multiclass',      # 目标函数: 'multiclass' 表示这是一个多分类问题。模型的目标是最小化这个函数定义的损失。
            'num_class': 3,                 # 类别数量: 必须与数据集中的类别数一致。这里是3，对应牛市、熊市、震荡市。
            'boosting_type': 'gbdt',        # 提升类型: 'gbdt' (Gradient Boosting Decision Tree) 是最经典、最常用的算法。
                                            # 其他选项如 'dart' (引入Dropout) 或 'goss' (基于梯度的单边采样) 可在特定场景下提升性能或速度，但 'gbdt' 是最稳健的起点。
            'metric': 'multi_logloss',      # 评估指标: 'multi_logloss' (多分类对数损失) 是多分类问题中常用的评估指标，
                                            # 用于在验证集上监控模型性能，并作为早停(early_stopping)的依据。
            # --- 性能与速度 (影响训练效率和模型学习能力) ---
            'n_estimators': 2000,           # 迭代次数 (树的数量): 模型要构建的树的总数。
                                            # 改大: 允许模型进行更多轮的学习，可能捕捉更复杂的模式，但增加训练时间。
                                            # 改小: 训练速度更快，但可能导致欠拟合。
                                            # 后果: 通常设置一个较大的值（如1000-10000），并配合早停使用，让模型在验证集性能不再提升时自动停止，从而找到最佳迭代次数。
            'learning_rate': 0.001,          # 学习率: 控制每次迭代（每棵树）对最终结果的贡献程度。
                                            # 改小 (如0.005): 模型学习得更慢、更精细，需要更多的 'n_estimators' 才能达到好的效果，但通常能找到泛化能力更好的模型。
                                            # 改大 (如0.1): 模型学习得更快，但容易跳过最优解，导致模型性能下降。
                                            # 后果: 'learning_rate' 和 'n_estimators' 强相关。推荐策略是：低 'learning_rate' (如 0.01-0.05) + 高 'n_estimators' + 早停。原值0.005过于保守，0.01是更均衡的起点。
            'n_jobs': -1,                   # 并行线程数: '-1' 表示使用所有可用的CPU核心，以最大化训练速度。
            'seed': 42,                     # 随机种子: 用于确保每次训练的结果可复现。只要种子不变，数据划分、特征选择等随机过程的结果就会固定。
            'verbose': -1,                  # 日志详细程度: '-1' 表示不输出训练过程中的详细日志，保持控制台输出干净。
            # --- 控制过拟合 (关键调优区，防止模型在新数据上表现差) ---
            'num_leaves': 5,               # 每棵树的最大叶子节点数: 控制模型复杂度的核心参数。
                                            # 改大: 允许树模型学习到更复杂的规则，可能提升训练集表现，但极易导致过拟合。
                                            # 改小: 限制树的复杂度，防止过拟合，但过小会使模型过于简单，导致欠拟合。
                                            # 后果: 它的值应小于 2^max_depth。原代码中的 '5' 过于保守，可能导致模型欠拟合。'31' 是一个常用的、稳健的默认值。
            'max_depth': 20,                 # 树的最大深度: 限制树可以生长的最大层数。
                                            # 改大: 允许树生长得更深，捕捉更具体的特征交互，但也增加了过拟合的风险。
                                            # 改小: 限制树的深度，是防止过拟合的有效手段。
                                            # 后果: 在LightGBM中，通常优先用 'num_leaves' 控制复杂度。设置 'max_depth' 可作为辅助手段，防止树长得过深。'-1'表示不限制。
            'subsample': 0.8,               # 样本采样比例 (行采样): 每次迭代时，从总训练数据中随机采样的比例。
                                            # 改小 (如0.7): 每次只用一部分数据训练，增加了模型的随机性，有助于防止过拟合，提高泛化能力。值必须在 (0, 1.0] 之间。
                                            # 改大 (如1.0): 每次都使用全部数据，失去了这个正则化效果。
                                            # 后果: 通常设置为 0.7-0.9 之间，可以有效防止过拟合。
            'colsample_bytree': 0.8,        # 特征采样比例 (列采样): 每次迭代时，从总特征中随机选择的比例。
                                            # 改小 (如0.7): 每次只用一部分特征来建树，有助于防止模型过度依赖少数几个强特征，从而防止过拟合。
                                            # 改大 (如1.0): 每次都考虑所有特征。
                                            # 后果: 和 'subsample' 类似，是重要的正则化手段。通常设置为 0.7-0.9 之间。
            'reg_alpha': 0.1,               # L1 正则化项: 对模型权重施加L1惩罚。
                                            # 改大: 使模型权重更趋向于0，可能产生稀疏模型（部分特征权重为0），有助于特征选择和防止过拟合。
                                            # 改小: 惩罚力度减弱。
            'reg_lambda': 0.1,              # L2 正则化项: 对模型权重施加L2惩罚。
                                            # 改大: 使模型权重整体变小但不会变为0，使模型更平滑，防止过拟合。
                                            # 改小: 惩罚力度减弱。
                                            # 后果: 这两个参数用于处理过拟合，特别是当特征数量很多时。通常从一个较小的值（如0.1）开始尝试。
        }


        # 步骤 3/4: 使用 Optuna 进行超参数优化和验证
        self.stdout.write("步骤 3/4: 使用 Optuna 进行超参数优化...")
        # 将数据的后20%作为验证集，前80%作为训练集
        split_point = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
        self.stdout.write(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        # 计算训练集样本权重
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )
        # 定义 Optuna 的目标函数 (objective function)
        def objective(trial):
            try:
                bull_label_index = [k for k, v in label_map.items() if v == 'Bull'][0]
                bear_label_index = [k for k, v in label_map.items() if v == 'Bear'][0]
            except IndexError:
                raise ValueError("在 label_map 中找不到 'Bull' 或 'Bear'。请检查数据集文件。")
            # --- 定义要搜索的参数空间 ---
            params_to_tune = {
                'num_leaves': trial.suggest_int('num_leaves', 2, 100),
                'max_depth': trial.suggest_int('max_depth', 1, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
            }
            
            # 合并基础参数和当前尝试的参数
            current_params = lgbm_params.copy()
            current_params.update(params_to_tune)
            # --- 使用时序交叉验证进行评估 ---
            # n_splits=5 表示创建5个 (训练集, 验证集) 对
            tscv = TimeSeriesSplit(n_splits=5) 
            scores = []
            for train_index, val_index in tscv.split(X):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                # 在每个折叠内部重新计算样本权重
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
                
                model = lgb.LGBMClassifier(**current_params)
                
                # 训练模型
                model.fit(X_train, y_train, sample_weight=sample_weights)
                # 在验证集上进行预测
                val_preds_class = model.predict(X_val)
                
                # --- 4. 计算并提取我们关心的F1分数 ---
                # average=None: 返回一个数组，包含每个类的F1分数
                # labels=...: 确保返回数组的顺序与我们的标签索引(0, 1, 2)一致
                f1_scores_per_class = f1_score(
                    y_val, 
                    val_preds_class, 
                    average=None, 
                    labels=sorted(label_map.keys()), 
                    zero_division=0
                )
                
                # 从F1分数数组中，根据索引提取牛市和熊市的分数
                bull_f1 = f1_scores_per_class[bull_label_index]
                bear_f1 = f1_scores_per_class[bear_label_index]
                
                # 计算这两个分数的平均值作为本次折叠的最终得分
                fold_score = (bull_f1 + bear_f1) / 2.0
                scores.append(fold_score)
            # 返回所有折叠分数的平均值，Optuna将优化这个平均值
            return np.mean(scores)
        # --- 运行 Optuna 优化 ---
        self.stdout.write("开始参数搜索 (n_trials=50)...")
        study = optuna.create_study(direction='maximize') # 目标是最小化 logloss
        study.optimize(objective, n_trials=500) # 尝试 50 组参数组合
        # --- 获取并报告最佳参数 ---
        # 合并基础参数和 Optuna 找到的最佳参数
        best_params = lgbm_params.copy()
        best_params.update(study.best_params)
        self.stdout.write(self.style.SUCCESS(f"\n搜索完成！最佳验证集 LogLoss: {study.best_value:.4f}"))
        self.stdout.write(self.style.SUCCESS(f"找到的最佳参数: {study.best_params}"))
        # --- 使用最佳参数在验证集上进行最终评估 ---
        self.stdout.write(self.style.SUCCESS("\n--- 使用最佳参数在验证集上进行评估 ---"))
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train,
                       sample_weight=sample_weights,
                       eval_set=[(X_val, y_val)],
                       eval_metric='multi_logloss',
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        val_preds_class = best_model.predict(X_val)
        self.stdout.write(classification_report(y_val, val_preds_class, target_names=target_names))
        self.stdout.write("\n--- 验证集混淆矩阵 ---")
        self.stdout.write(str(confusion_matrix(y_val, val_preds_class)))
        # 步骤 4/4: 使用最佳参数训练最终模型并保存
        self.stdout.write("\n步骤 4/4: 使用最佳参数训练最终模型并保存...")
        # 训练最终模型 (在所有数据上)
        self.stdout.write("\n--- 训练最终模型 (使用全部数据和最佳参数) ---")
        final_model = lgb.LGBMClassifier(**best_params)
        final_sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        
        # 注意：在最终模型上训练时，不再需要 eval_set 和 early_stopping
        final_model.fit(X, y, sample_weight=final_sample_weights)
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
        # 更新模型配置文件，加入特征重要性
        try:
            with open(self.MODEL_CONFIG_FILE, 'r') as f:
                model_config = json.load(f)
            
            model_config['feature_importance'] = feature_importance_df.to_dict('records')
            # 同时可以保存最佳参数以供参考
            model_config['best_params'] = study.best_params
            with open(self.MODEL_CONFIG_FILE, 'w') as f:
                json.dump(model_config, f, indent=4)
            self.stdout.write(self.style.SUCCESS(f"特征重要性和最佳参数已更新至: {self.MODEL_CONFIG_FILE}"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"更新模型配置文件时出错: {e}"))
        self.stdout.write(self.style.SUCCESS("===== [M-Value Refactor] 模型训练流程结束！ ====="))
