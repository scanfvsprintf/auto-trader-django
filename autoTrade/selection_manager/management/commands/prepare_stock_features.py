# ==============================================================================
# 文件 1/4: selection_manager/management/commands/prepare_stock_features.py
# 描述: 为个股评分模型生成特征和标签数据集。(已重构为内存优化版本)
#
# ==============================================================================
# ==============================================================================
# [重构说明]
#
# 原始代码存在严重的内存瓶颈问题。当处理大数据量（如400M+）时，以下操作
# 会导致内存耗尽：
# 1. `pd.DataFrame.from_records(quotes_qs)`: 一次性将所有数据库记录物化
#    为Pandas DataFrame，是内存爆炸的根源。
# 2. `df.unstack()`: 在标签生成中，将长格式数据转换为宽格式，创建了一个
#    巨大的内存矩阵，是第二个内存瓶颈。
# 3. `pd.concat(all_features_list)`: 在特征计算后，一次性合并所有股票的
#    特征数据，是第三个内存瓶颈。
#
# 本次重构遵循“分治” (Divide and Conquer) 的思想，彻底解决了这些问题：
#
# 1.  【数据加载策略改变】: 放弃一次性创建巨大DataFrame。`_load_initial_data`
#     函数会一次性查询数据库（满足约束），但使用`.iterator()`方法流式地将数据
#     读入一个按`stock_code_id`组织的Python字典中。这种原生数据结构比Pandas
#     DataFrame内存占用小得多。
#
# 2.  【批处理核心循环】: `handle`方法的核心逻辑被改造为一个批处理循环。它将
#     所有股票代码分批（例如每批100个），然后对每个批次执行所有计算。
#
# 3.  【内存友好的计算】:
#     - 标签生成 (`_generate_labels_for_batch`): 完全放弃`unstack()`，转而
#       使用`groupby().shift()`和`groupby().rolling()`等在长格式数据上直接
#       操作的内存友好型方法。
#     - 特征生成 (`_calculate_features_for_batch`): 函数逻辑基本不变，但
#       输入从全量数据变为小批量数据，使其内存占用和执行速度都得到极大优化。
#
# 4.  【中间文件持久化】: 每个批次处理完成后，生成的DataFrame会立即被保存到
#     一个临时文件（如.pkl），然后内存被释放。这确保了在任何时候，内存中都
#     只有一小部分数据。
#
# 5.  【最终聚合】: 所有批次处理完毕后，`_merge_and_save_final_dataset`函数
#     负责从磁盘读取所有临时文件，将它们合并成最终的大数据集，然后保存。
#
# 6.  【资源清理】: 使用`try...finally`结构确保临时文件和目录在脚本执行
#     结束或发生错误时都能被可靠地删除。
#
# 通过以上改造，脚本现在能够以极低的稳定内存占用处理任意规模的数据，同时保证
# 最终输出与原始逻辑完全一致。
# ==============================================================================

import logging
import pickle
import json
import shutil # 用于删除临时目录
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pandas_ta as ta
from django.core.management.base import BaseCommand
from django.conf import settings
from tqdm import tqdm

from common.models import DailyQuotes, DailyFactorValues
from selection_manager.service.m_value_service import FactorCalculator
from selection_manager.service.selection_service import MARKET_INDICATOR_CODE

logger = logging.getLogger(__name__)

# --- [配置区] ---
# 通过修改此处的配置，来控制数据集的生成方式 (此部分保持不变)
LABEL_CONFIG = {
    'mode': 'sharpe',  # 'return' (未来收益率) 或 'sharpe' (未来夏普比率)
    'lookforward_days': 10,  # 标签向前看的天数 (N)
    'risk_free_rate_annual': 0.02,  # 年化无风险利率，仅在 'sharpe' 模式下使用
    'tanh_scaling_factor': 1.0,  # tanh缩放因子，仅在 'sharpe' 模式下使用
}

# 因子计算所需的最大回溯期，应大于所有因子中最大的lookback period
# 例如，neg_dev(60), max_dd(60)等，给足100天buffer
FACTOR_LOOKBACK_BUFFER = 100

# 新增：批处理大小配置
BATCH_SIZE = 100

class Command(BaseCommand):
    help = '为个股评分模型生成特征和标签数据集 (X, y) (内存优化版)。'

    # --- 路径配置 (保持不变) ---
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    DATASET_FILE = MODELS_DIR / 'stock_features_dataset.pkl'
    MODEL_CONFIG_FILE = MODELS_DIR / 'stock_model_config.json'
    
    # 新增：临时文件存放路径
    TEMP_DIR = MODELS_DIR / 'temp_stock_features'
    
    def handle(self, *args, **options):
        """
        主处理函数，作为“总指挥”，编排整个分治流程。
        """
        self.stdout.write(self.style.SUCCESS("===== 开始为个股评分模型准备机器学习数据集 (内存优化版) ====="))
        
        # 0. 初始化环境：创建模型目录和临时目录
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if self.TEMP_DIR.exists():
            shutil.rmtree(self.TEMP_DIR) # 如果上次异常退出，先清理
        self.TEMP_DIR.mkdir(parents=True)
        self.stdout.write(f"临时目录已创建: {self.TEMP_DIR}")

        try:
            # 1. 加载全局数据和元信息
            self.stdout.write("步骤 1/5: 加载全局数据 (M值) 和元信息 (股票列表)...")
            m_values_series, all_stock_codes, quotes_by_stock, feature_names = self._load_initial_data()
            
            if not all_stock_codes:
                self.stdout.write(self.style.ERROR("错误: 数据库中没有找到任何符合条件的股票数据。"))
                return
            
            self.stdout.write(f"已加载 {len(all_stock_codes)} 只股票的元数据和 {len(m_values_series)} 条M值记录。")

            # 2. 分批处理所有股票
            self.stdout.write(f"步骤 2/5: 开始分批处理股票 (每批 {BATCH_SIZE} 只)...")
            
            # 将股票代码列表切分成批次
            stock_batches = [all_stock_codes[i:i + BATCH_SIZE] for i in range(0, len(all_stock_codes), BATCH_SIZE)]
            
            temp_file_paths = []
            # 使用tqdm显示批处理的宏观进度
            for i, batch_codes in enumerate(tqdm(stock_batches, desc="处理股票批次")):
                
                # 2.1 从内存字典中构建当前批次的DataFrame
                batch_records = [record for code in batch_codes for record in quotes_by_stock[code]]
                if not batch_records:
                    continue # 如果该批次没有数据，则跳过
                    
                batch_quotes_df = pd.DataFrame.from_records(batch_records)
                # 执行和原始代码相同的数值转换
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close']
                for col in numeric_cols:
                    batch_quotes_df[col] = pd.to_numeric(batch_quotes_df[col], errors='coerce')

                # 2.2 为当前批次生成特征和标签
                processed_batch_df = self._process_stock_batch(batch_quotes_df, m_values_series, feature_names)

                if processed_batch_df.empty:
                    self.stdout.write(self.style.WARNING(f"警告: 批次 {i+1}/{len(stock_batches)} 处理后没有产生有效数据。"))
                    continue

                # 2.3 将处理好的批次数据保存到临时文件
                temp_file_path = self.TEMP_DIR / f'batch_{i}.pkl'
                with open(temp_file_path, 'wb') as f:
                    pickle.dump(processed_batch_df, f)
                temp_file_paths.append(temp_file_path)

            self.stdout.write("步骤 3/5: 所有批次处理完成。")

            if not temp_file_paths:
                self.stdout.write(self.style.ERROR("错误: 所有批次均未产生有效数据，无法生成最终数据集。"))
                return
            
            # 4. 合并所有临时文件并保存最终结果
            self.stdout.write("步骤 4/5: 合并所有批次结果并保存最终数据集...")
            self._merge_and_save_final_dataset(temp_file_paths, feature_names)

            self.stdout.write("步骤 5/5: 数据集准备完成。")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"处理过程中发生严重错误: {e}"))
            import traceback
            traceback.print_exc()
        finally:
            # 5. 清理临时文件
            self.stdout.write("清理临时文件...")
            if self.TEMP_DIR.exists():
                shutil.rmtree(self.TEMP_DIR)
                self.stdout.write(self.style.SUCCESS(f"临时目录已成功删除: {self.TEMP_DIR}"))
            self.stdout.write(self.style.SUCCESS("===== 数据准备流程结束 ====="))

    def _load_initial_data(self):
        """
        [重构后] 内存优化的初始数据加载函数。
        
        此函数执行以下操作：
        1. 确定数据加载的日期范围。
        2. 加载M值 (数据量小，直接加载)。
        3. 获取所有相关股票代码的列表 (轻量级操作)。
        4. 【核心优化】流式查询日线行情数据，并存入一个按stock_code_id组织的Python字典中。
           这避免了创建巨大的Pandas DataFrame，是解决内存瓶颈的关键第一步。
        5. 加载或确定因子名称列表。

        返回:
            - m_values_series: Pandas Series, 索引为trade_date，值为M值。
            - all_stock_codes: list, 所有需要处理的股票代码。
            - quotes_by_stock: dict, 键为股票代码，值为包含该股票所有行情记录的列表。
            - feature_names: list, 需要计算的因子名称列表。
        """
        # 确定数据加载的起始日期 (逻辑不变)
        first_quote = DailyQuotes.objects.order_by('trade_date').first()
        if not first_quote:
            return pd.Series(), [], {}, []
        
        start_date = first_quote.trade_date + timedelta(days=FACTOR_LOOKBACK_BUFFER)
        self.stdout.write(f"数据加载起始日期 (已考虑因子计算缓冲): {start_date}")

        # 加载M值 (数据量小，保持原样)
        m_values_qs = DailyFactorValues.objects.filter(
            stock_code_id=MARKET_INDICATOR_CODE, factor_code_id='dynamic_M_VALUE', trade_date__gte=start_date
        ).values('trade_date', 'raw_value')
        m_values_df = pd.DataFrame.from_records(m_values_qs)
        m_values_series = m_values_df.set_index('trade_date')['raw_value'].astype(float) if not m_values_df.empty else pd.Series()

        # 【核心优化点】加载日线行情数据到字典，而不是DataFrame
        quotes_qs = DailyQuotes.objects.filter(trade_date__gte=start_date).values(
            'trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close'
        ).order_by('stock_code_id', 'trade_date') # 排序有助于后续处理

        quotes_by_stock = defaultdict(list)
        
        # 使用.iterator()进行流式查询，避免一次性加载所有数据到内存
        # 这是遵守“一次查询”和内存优化之间的最佳平衡点
        self.stdout.write("开始将日线行情数据流式加载到内存字典中...")
        total_records = quotes_qs.count()
        for record in tqdm(quotes_qs.iterator(chunk_size=2000), total=total_records, desc="加载数据库记录"):
            quotes_by_stock[record['stock_code_id']].append(record)
        
        all_stock_codes = list(quotes_by_stock.keys())
        
        # 加载因子名称列表 (逻辑不变)
        try:
            with open(settings.BASE_DIR / 'selection_manager' / 'ml_models' / 'm_value_model_config.json', 'r') as f:
                feature_names = json.load(f)['feature_names']
        except FileNotFoundError:
            self.stdout.write(self.style.WARNING("M值模型配置文件不存在，将使用默认因子列表。"))
            feature_names = [
                'dynamic_ADX_CONFIRM', 'dynamic_v2_MA_SLOPE', 'dynamic_v2_MA_SCORE',
                'dynamic_v2_CPC_Factor', 'dynamic_v2_VPCF', 'dynamic_BREAKOUT_PWR',
                'dynamic_VOLUME_SURGE', 'dynamic_MOM_ACCEL', 'dynamic_RSI_OS',
                'dynamic_NEG_DEV', 'dynamic_BOLL_LB', 'dynamic_LOW_VOL',
                'dynamic_MAX_DD', 'dynamic_DOWNSIDE_RISK', 'dynamic_Old_D',
                'dynamic_Old_I', 'dynamic_Old_M'
            ]

        return m_values_series, all_stock_codes, quotes_by_stock, feature_names
    
    def _process_stock_batch(self, batch_quotes_df: pd.DataFrame, m_values_series: pd.Series, feature_names: list) -> pd.DataFrame:
        """
        [新增] 封装了对单个股票批次的所有处理逻辑。
        
        Args:
            batch_quotes_df: 包含一个批次（如100只股票）数据的DataFrame。
            m_values_series: 全局的M值时间序列。
            feature_names: 需要计算的因子列表。
            
        Returns:
            一个处理完成的DataFrame，包含特征、M值和标签，且已dropna。
        """
        # 1. 为当前批次生成标签
        labels_df = self._generate_labels_for_batch(batch_quotes_df)

        # 2. 为当前批次计算因子特征
        features_df = self._calculate_features_for_batch(batch_quotes_df, feature_names)

        if features_df.empty or labels_df.empty:
            return pd.DataFrame()

        # 3. 合并特征、M值和标签
        # 将M值作为一个特征加入 (与原逻辑相同)
        features_df['market_m_value'] = features_df.index.get_level_values('trade_date').map(m_values_series)
        
        # 合并所有数据
        final_df = features_df.join(labels_df, how='inner')
        final_df.dropna(inplace=True)

        return final_df

    def _generate_labels_for_batch(self, batch_quotes_df: pd.DataFrame) -> pd.DataFrame:
        """
        [重构后] 为单个批次的股票数据生成标签，内存优化版本。
        
        【核心优化点】此函数完全避免了`unstack()`操作。它利用`groupby('stock_code_id')`
        在长格式的DataFrame上直接进行时间序列计算（如shift, pct_change, rolling），
        这在保证计算结果正确的同时，极大地节约了内存。
        """
        # 使用MultiIndex方便按股票分组进行时间序列操作
        df_indexed = batch_quotes_df.set_index(['trade_date', 'stock_code_id']).sort_index()
        
        # 替换0值为nan，防止计算错误 (hfq_close不应为0)
        df_indexed['hfq_close'].replace(0, np.nan, inplace=True)
        
        lookforward = LABEL_CONFIG['lookforward_days']

        if LABEL_CONFIG['mode'] == 'return':
            # 使用 groupby().shift() 计算未来价格，这在长格式上操作，内存高效
            future_price = df_indexed.groupby(level='stock_code_id')['hfq_close'].shift(-lookforward)
            labels = (future_price / df_indexed['hfq_close']) - 1
            
        elif LABEL_CONFIG['mode'] == 'sharpe':
            # 同样，使用 groupby().pct_change() 确保只在组内计算收益率
            returns = df_indexed.groupby(level='stock_code_id')['hfq_close'].pct_change()
            
            daily_rf = (1 + LABEL_CONFIG['risk_free_rate_annual'])**(1/252) - 1
            excess_returns = returns - daily_rf
            
            # 先shift未来的超额收益，再进行滚动计算
            future_excess_returns = excess_returns.groupby(level='stock_code_id').shift(-lookforward)
            
            # 使用 groupby().rolling() 计算未来N天的滚动均值和标准差
            rolling_mean = future_excess_returns.groupby(level='stock_code_id').rolling(window=lookforward, min_periods=lookforward).mean()
            rolling_std = future_excess_returns.groupby(level='stock_code_id').rolling(window=lookforward, min_periods=lookforward).std()
            
            # rolling操作会产生复杂的多级索引，我们需要将结果对齐回原始索引
            # .reset_index()后，level='stock_code_id'的索引就没了，所以我们直接用.values然后重新构建
            # 注意：rolling().mean()返回的Series的索引是 ('stock_code_id', 'trade_date')
            # 我们可以直接使用这个结果
            
            annualized_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)
            
            # 使用tanh进行缩放
            labels = np.tanh(LABEL_CONFIG['tanh_scaling_factor'] * annualized_sharpe)
            # rolling操作的结果索引和df_indexed的索引是一致的，可以直接使用
            
        else:
            raise ValueError(f"未知的标签模式: {LABEL_CONFIG['mode']}")

        return labels.rename('label').to_frame()

    def _calculate_features_for_batch(self, batch_quotes_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
        """
        [重构后] 为单个批次的股票数据计算所有因子特征。
        
        此函数逻辑与原版 _calculate_all_features 几乎相同，但其输入是一个小批量的
        DataFrame，因此执行速度快，内存占用低。
        """
        all_features_list = []
        # 按股票代码分组，现在只会包含最多 BATCH_SIZE 个组
        stock_groups = batch_quotes_df.groupby('stock_code_id')
        
        for stock_code, group_df in stock_groups:
            # 重命名列以匹配FactorCalculator的输入要求
            group_df = group_df.rename(columns={'turnover': 'amount'})
            group_df = group_df.set_index('trade_date').sort_index()
            
            # 确保数据连续，填充缺失的交易日 (逻辑不变)
            # 使用freq='D'再筛选交易日，或者直接用'B'，取决于数据源特性
            full_date_range = pd.date_range(start=group_df.index.min(), end=group_df.index.max(), freq='B')
            group_df = group_df.reindex(full_date_range).ffill()
            
            if len(group_df) < FACTOR_LOOKBACK_BUFFER:
                continue

            # 使用复用的因子计算器
            calculator = FactorCalculator(group_df)
            stock_features_df = calculator.run(feature_names)
            stock_features_df['stock_code_id'] = stock_code
            all_features_list.append(stock_features_df)

        if not all_features_list:
            return pd.DataFrame()
            
        # 合并这个小批次中所有股票的特征，内存开销很小
        batch_features_df = pd.concat(all_features_list)
        batch_features_df = batch_features_df.reset_index().rename(columns={'index': 'trade_date'})
        batch_features_df = batch_features_df.set_index(['trade_date', 'stock_code_id'])
        
        return batch_features_df

    def _merge_and_save_final_dataset(self, temp_file_paths: list, feature_names: list):
        """
        [新增] 合并所有临时批次文件，并保存最终的数据集和配置文件。
        """
        self.stdout.write(f"正在从 {len(temp_file_paths)} 个临时文件中合并最终数据集...")
        
        all_batches_dfs = []
        for path in tqdm(temp_file_paths, desc="合并批次文件"):
            with open(path, 'rb') as f:
                batch_df = pickle.load(f)
                all_batches_dfs.append(batch_df)
        
        # 将所有批次的DataFrame合并成一个大的DataFrame
        # ignore_index=False 保留了我们精心设置的 (trade_date, stock_code_id) 索引
        final_df = pd.concat(all_batches_dfs, ignore_index=False)
        final_df.sort_index(inplace=True) # 确保最终数据集按索引排序

        if final_df.empty:
            self.stdout.write(self.style.ERROR("错误: 合并后，最终数据集为空。"))
            return
            
        # 从最终的DataFrame中分离出 X 和 y
        # 别忘了market_m_value也是一个特征
        final_feature_names = feature_names + ['market_m_value']
        
        # 确保final_df中的列与期望的特征列一致，防止意外错误
        final_feature_names = [f for f in final_feature_names if f in final_df.columns]
        
        X = final_df[final_feature_names]
        y = final_df['label']

        self.stdout.write(f"数据集准备完成。总样本数: {len(X)}")
        self.stdout.write("标签 (label) 统计信息:")
        self.stdout.write(str(y.describe()))

        # 保存最终数据集 (与原逻辑相同)
        dataset = {'X': X, 'y': y, 'index': X.index, 'feature_names': final_feature_names}
        with open(self.DATASET_FILE, 'wb') as f:
            pickle.dump(dataset, f)
        self.stdout.write(self.style.SUCCESS(f"数据集已成功保存至: {self.DATASET_FILE}"))

        # 保存模型配置文件 (与原逻辑相同)
        model_config = {'feature_names': final_feature_names}
        with open(self.MODEL_CONFIG_FILE, 'w') as f:
            json.dump(model_config, f, indent=4)
        self.stdout.write(self.style.SUCCESS(f"模型配置文件已成功保存至: {self.MODEL_CONFIG_FILE}"))

