# ==============================================================================
# 文件 1/4: selection_manager/management/commands/prepare_stock_features.py
# 描述: 为个股评分模型生成特征和标签数据集。(已改造为分批处理)
# ==============================================================================
import logging
import pickle
import json
import shutil # 导入shutil库用于删除目录
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
from django.core.management.base import BaseCommand
from django.conf import settings
from tqdm import tqdm

from common.models import DailyQuotes, DailyFactorValues
from selection_manager.service.m_value_service import FactorCalculator # 复用M值服务中的因子计算器
from selection_manager.service.selection_service import MARKET_INDICATOR_CODE

logger = logging.getLogger(__name__)

# --- [配置区] ---
# 通过修改此处的配置，来控制数据集的生成方式
LABEL_CONFIG = {
    'mode': 'sharpe',  # 'return' (未来收益率) 或 'sharpe' (未来夏普比率)
    'lookforward_days': 3, # 标签向前看的天数 (N)
    'risk_free_rate_annual': 0.02, # 年化无风险利率，仅在 'sharpe' 模式下使用
    'tanh_scaling_factor': 1.0, # tanh缩放因子，仅在 'sharpe' 模式下使用
}

# 因子计算所需的最大回溯期，应大于所有因子中最大的lookback period
# 例如，neg_dev(60), max_dd(60)等，给足100天buffer
FACTOR_LOOKBACK_BUFFER = 100

# --- [新增配置区] ---
# 分批处理的配置
BATCH_SIZE = 100 # 每次处理100只股票，以控制内存使用

class Command(BaseCommand):
    help = '为个股评分模型生成特征和标签数据集 (X, y)。采用分批处理以优化内存。'
    def add_arguments(self, parser):
        """添加命令行参数"""
        parser.add_argument(
            '--use-local-db',
            action='store_true',  # 这使它成为一个开关，存在即为True
            help='如果指定，则将数据源切换至 D:\\project\\mainDB.sqlite3 (需在settings.py中配置好local_sqlite)。'
        )
    # --- 路径配置 ---
    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    DATASET_FILE = MODELS_DIR / 'stock_features_dataset.pkl'
    MODEL_CONFIG_FILE = MODELS_DIR / 'stock_model_config.json' # 与个股模型共享配置
    
    # --- [新增] 临时文件存储路径 ---
    TEMP_DATA_DIR = MODELS_DIR / 'temp_feature_batches'


    def handle(self, *args, **options):
        """
        主处理函数，协调整个数据集的生成流程。
        改造后的流程：
        1. 全局准备：获取所有股票代码、M值、因子名等一次性数据。
        2. 分批处理：将股票代码分批，对每一批次执行“加载-计算-合并”并存为临时文件。
        3. 最终合并：将所有临时文件合并为最终数据集。
        4. 清理：删除临时文件。
        """

        use_local_db = options['use_local_db']
        self.db_alias = 'local_sqlite' if use_local_db else 'default'
        
        db_source_message = f"D:\\project\\mainDB.sqlite3 (别名: {self.db_alias})" if use_local_db else f"默认数据库 (别名: {self.db_alias})"
        self.stdout.write(self.style.SUCCESS(f"当前使用的数据源: {db_source_message}"))

        self.stdout.write(self.style.SUCCESS("===== 开始为个股评分模型准备机器学习数据集 (分批处理模式) ====="))
        
        # --- [改造] 步骤 0: 准备工作，创建临时目录 ---
        # 如果临时目录存在，先删除，确保一个干净的开始
        if self.TEMP_DATA_DIR.exists():
            shutil.rmtree(self.TEMP_DATA_DIR)
        self.TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.stdout.write(f"创建临时目录: {self.TEMP_DATA_DIR}")

        try:
            # --- [改造] 步骤 1: 一次性加载全局/轻量级数据 ---
            self.stdout.write("步骤 1/5: 加载全局数据 (股票列表, M值)...")
            all_stock_codes, m_values_series, start_date = self._load_global_data()
            if not all_stock_codes:
                self.stdout.write(self.style.ERROR("错误: 数据库中没有可用的股票代码。"))
                return
            
            # 从M值模型配置中获取所有需要计算的因子名称
            feature_names = self._get_feature_names()

            # --- [改造] 步骤 2: 分批生成特征和标签，并保存为临时文件 ---
            self.stdout.write(f"步骤 2/5: 开始分批处理 {len(all_stock_codes)} 只股票，每批 {BATCH_SIZE} 只...")
            
            # 将所有股票代码分批
            stock_batches = [all_stock_codes[i:i + BATCH_SIZE] for i in range(0, len(all_stock_codes), BATCH_SIZE)]
            
            # 使用tqdm显示批次处理进度
            batch_iterator = tqdm(enumerate(stock_batches), total=len(stock_batches), desc="处理批次")
            
            intermediate_files = [] # 用于存储生成的中间文件名

            for i, batch_codes in batch_iterator:
                batch_iterator.set_description(f"处理批次 {i+1}/{len(stock_batches)}")
                self.stdout.write(f"\n  - [批次 {i+1}] 开始加载 {len(batch_codes)} 只股票的行情数据...")
                # 2.1 加载当前批次的行情数据
                quotes_df = self._load_batch_quotes(batch_codes, start_date)
                if quotes_df.empty:
                    self.stdout.write(self.style.WARNING(f"警告: 批次 {i+1} 未能加载到任何数据，已跳过。"))
                    continue
                self.stdout.write(f"  - [批次 {i+1}] 行情数据加载完成，共 {len(quotes_df)} 条记录。开始生成标签...")
                # 2.2 为当前批次生成标签
                labels_df = self._generate_labels(quotes_df)
                self.stdout.write(f"  - [批次 {i+1}] 标签生成完成。开始计算特征...")
                # 2.3 为当前批次计算因子特征
                features_df = self._calculate_batch_features(quotes_df, feature_names)
                self.stdout.write(f"  - [批次 {i+1}] 标签生成完成。开始计算特征...")
                # 2.4 合并批次内的特征、M值和标签
                # 将M值作为一个特征加入
                features_df['market_m_value'] = features_df.index.get_level_values('trade_date').map(m_values_series)
                
                # 合并批次数据
                batch_final_df = features_df.join(labels_df, how='inner')
                batch_final_df.dropna(inplace=True)

                if batch_final_df.empty:
                    continue

                # 2.5 将处理好的批次数据保存到临时文件
                temp_file_path = self.TEMP_DATA_DIR / f'batch_{i}.pkl'
                with open(temp_file_path, 'wb') as f:
                    pickle.dump(batch_final_df, f)
                intermediate_files.append(temp_file_path)

            if not intermediate_files:
                self.stdout.write(self.style.ERROR("错误: 所有批次处理后未生成任何有效数据。请检查数据范围或因子计算。"))
                return

            # --- [改造] 步骤 3 & 4: 合并所有临时文件 ---
            self.stdout.write(f"\n步骤 3/5 & 4/5: 合并 {len(intermediate_files)} 个中间文件...")
            all_dfs = []
            for file_path in tqdm(intermediate_files, desc="合并文件"):
                with open(file_path, 'rb') as f:
                    batch_df = pickle.load(f)
                    all_dfs.append(batch_df)
            
            final_df = pd.concat(all_dfs, ignore_index=False) # ignore_index=False保留(date, code)多级索引

            # --- [无改动] 步骤 5: 保存最终数据集和配置 ---
            self.stdout.write("步骤 5/5: 保存最终数据集和模型配置文件...")
            # 在最终合并后的DataFrame中加入market_m_value特征名
            if 'market_m_value' not in feature_names:
                feature_names.append('market_m_value')

            X = final_df[feature_names]
            y = final_df['label']

            self.stdout.write(f"数据集准备完成。总样本数: {len(X)}")
            self.stdout.write("标签 (label) 统计信息:")
            self.stdout.write(str(y.describe()))

            dataset = {'X': X, 'y': y, 'index': X.index, 'feature_names': feature_names}
            with open(self.DATASET_FILE, 'wb') as f:
                pickle.dump(dataset, f)
            self.stdout.write(self.style.SUCCESS(f"数据集已成功保存至: {self.DATASET_FILE}"))

            model_config = {'feature_names': feature_names}
            with open(self.MODEL_CONFIG_FILE, 'w') as f:
                json.dump(model_config, f, indent=4)
            self.stdout.write(self.style.SUCCESS(f"模型配置文件已成功保存至: {self.MODEL_CONFIG_FILE}"))
            
        finally:
            # --- [新增] 清理步骤 ---
            # 无论成功与否，都尝试删除临时目录
            if self.TEMP_DATA_DIR.exists():
                shutil.rmtree(self.TEMP_DATA_DIR)
                self.stdout.write(self.style.SUCCESS(f"临时目录已清理: {self.TEMP_DATA_DIR}"))
        
        self.stdout.write(self.style.SUCCESS("===== 数据准备流程结束 ====="))

    def _load_global_data(self):
        """
        [新增] 一次性从数据库加载全局共享且数据量较小的数据。
        - 所有唯一的股票代码列表。
        - 市场M值时间序列。
        - 计算并返回全局的数据起始日期。
        """
        # 确定数据加载的起始日期
        first_quote = DailyQuotes.objects.order_by('trade_date').first()
        if not first_quote:
            return [], pd.Series(), None
        
        start_date = first_quote.trade_date + timedelta(days=FACTOR_LOOKBACK_BUFFER)
        self.stdout.write(f"全局数据加载起始日期 (已考虑因子计算缓冲): {start_date}")

        # 加载所有唯一的股票代码
        all_stock_codes = list(DailyQuotes.objects.using(self.db_alias).values_list('stock_code_id', flat=True).distinct())
        
        # 加载M值
        m_values_qs = DailyFactorValues.objects.filter(
            stock_code_id=MARKET_INDICATOR_CODE,
            factor_code_id='dynamic_M_VALUE',
            trade_date__gte=start_date
        ).values('trade_date', 'raw_value')
        m_values_df = pd.DataFrame.from_records(m_values_qs)
        m_values_series = pd.Series()
        if not m_values_df.empty:
            m_values_series = m_values_df.set_index('trade_date')['raw_value'].astype(float)
        
        return all_stock_codes, m_values_series, start_date
    
    def _load_batch_quotes(self, batch_codes: list, start_date):
        """
        [新增] 从数据库加载一个批次股票的日线行情数据。
        """
        # 加载指定股票批次的日线行情
        quotes_qs = DailyQuotes.objects.using(self.db_alias).filter(
            trade_date__gte=start_date,
            stock_code_id__in=batch_codes # 核心改动：只查询当前批次的股票
        ).values(
            'trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close'
        )
        if not quotes_qs.exists():
            return pd.DataFrame()

        quotes_df = pd.DataFrame.from_records(quotes_qs)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close']
        for col in numeric_cols:
            quotes_df[col] = pd.to_numeric(quotes_df[col], errors='coerce')
        
        quotes_df['adj_factor'] = quotes_df['hfq_close'] / (quotes_df['close'] + 1e-9)
        
        # 3. 计算后复权的 open, high, low
        quotes_df['hfq_open'] = quotes_df['open'] * quotes_df['adj_factor']
        quotes_df['hfq_high'] = quotes_df['high'] * quotes_df['adj_factor']
        quotes_df['hfq_low'] = quotes_df['low'] * quotes_df['adj_factor']
        
        # 4. 为了后续计算器接口统一，重命名列
        #    现在 'open', 'high', 'low', 'close' 都代表后复权价格
        final_df = pd.DataFrame({
            'trade_date': quotes_df['trade_date'],
            'stock_code_id': quotes_df['stock_code_id'],
            'open': quotes_df['hfq_open'],
            'high': quotes_df['hfq_high'],
            'low': quotes_df['hfq_low'],
            'close': quotes_df['hfq_close'], # 直接使用hfq_close作为复权收盘价
            'volume': quotes_df['volume'],
            'amount': quotes_df['turnover'] # 同时在这里完成turnover到amount的重命名
        })
        
        return final_df


    def _generate_labels(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """
        [无改动] 根据配置为给定的DataFrame生成标签。
        此函数逻辑不变，现在作用于小批量的DataFrame，因此内存友好。
        """
        df = quotes_df.set_index(['trade_date', 'stock_code_id'])['close'].unstack()
        df.replace(0, np.nan, inplace=True)
        if LABEL_CONFIG['mode'] == 'return':
            # 计算未来N日收益率
            future_price = df.shift(-LABEL_CONFIG['lookforward_days'])
            labels = (future_price / df) - 1
        elif LABEL_CONFIG['mode'] == 'sharpe':
            # 计算未来N日夏普比率
            returns = df.pct_change(fill_method=None)
            daily_rf = (1 + LABEL_CONFIG['risk_free_rate_annual'])**(1/252) - 1
            excess_returns = returns - daily_rf
            
            future_mean = excess_returns.shift(-LABEL_CONFIG['lookforward_days']).rolling(window=LABEL_CONFIG['lookforward_days']).mean()
            future_std = excess_returns.shift(-LABEL_CONFIG['lookforward_days']).rolling(window=LABEL_CONFIG['lookforward_days']).std()
            
            annualized_sharpe = (future_mean / future_std.replace(0, np.nan)) * np.sqrt(252)
            labels = np.tanh(LABEL_CONFIG['tanh_scaling_factor'] * annualized_sharpe)
        else:
            raise ValueError(f"未知的标签模式: {LABEL_CONFIG['mode']}")

        return labels.stack().rename('label').to_frame()

    def _get_feature_names(self):
        """
        [新增] 封装了获取因子名称列表的逻辑，只执行一次。
        """
        try:
            with open(settings.BASE_DIR / 'selection_manager' / 'ml_models' / 'm_value_model_config.json', 'r') as f:
                m_value_config = json.load(f)
            feature_names = m_value_config['feature_names']
            self.stdout.write("成功从'm_value_model_config.json'加载因子列表。")
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR("错误: M值模型配置文件 'm_value_model_config.json' 不存在。"))
            # 如果M值模型不存在，则使用一个预定义的因子列表
            feature_names = [
                'dynamic_ADX_CONFIRM', 'dynamic_v2_MA_SLOPE', 'dynamic_v2_MA_SCORE',
                'dynamic_v2_CPC_Factor', 'dynamic_v2_VPCF', 'dynamic_BREAKOUT_PWR',
                'dynamic_VOLUME_SURGE', 'dynamic_MOM_ACCEL', 'dynamic_RSI_OS',
                'dynamic_NEG_DEV', 'dynamic_BOLL_LB', 'dynamic_LOW_VOL',
                'dynamic_MAX_DD', 'dynamic_DOWNSIDE_RISK', 'dynamic_Old_D',
                'dynamic_Old_I', 'dynamic_Old_M'
            ]
            self.stdout.write(self.style.WARNING(f"将使用默认的因子列表: {feature_names}"))
        return feature_names

    def _calculate_batch_features(self, quotes_df: pd.DataFrame, feature_names: list):
        """
        [名称和签名有改动] 对一个批次的股票计算所有因子特征。
        原_calculate_all_features，逻辑不变，但现在处理的是小批量的DataFrame。
        """
        all_features_list = []
        stock_groups = quotes_df.groupby('stock_code_id')
        
        # 这里的tqdm现在显示单批次内的股票计算进度，可以禁用以减少日志输出
        for stock_code, group_df in stock_groups: # tqdm(stock_groups, desc="计算批次内因子", leave=False)
            #group_df = group_df.rename(columns={'turnover': 'amount'})
            group_df = group_df.set_index('trade_date').sort_index()
            
            # 确保数据连续，填充缺失的交易日
            min_date, max_date = group_df.index.min(), group_df.index.max()
            if pd.isna(min_date) or pd.isna(max_date):
                continue
            full_date_range = pd.date_range(start=min_date, end=max_date, freq='B')
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
            
        # 合并批次中所有股票的特征
        final_features_df = pd.concat(all_features_list)
        final_features_df = final_features_df.reset_index().rename(columns={'index': 'trade_date'})
        final_features_df = final_features_df.set_index(['trade_date', 'stock_code_id'])
        
        return final_features_df

