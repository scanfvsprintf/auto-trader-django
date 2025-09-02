# selection_manager/management/commands/prepare_stock_features.py
# [修改后]
# 描述: 为个股评分模型生成特征和标签数据集。采用分批处理以优化内存，
#       并集成了新的数据筛选、特征工程和标签工程逻辑。

import logging
import pickle
import json
import shutil
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings
from django.db.models import Min
from tqdm import tqdm

from common.models import DailyQuotes, DailyFactorValues, StockInfo, CorporateAction
from selection_manager.service.selection_service import MARKET_INDICATOR_CODE
# [修改] 导入新的因子计算器
from selection_manager.service.stock_factor_calculator import StockFactorCalculator

logger = logging.getLogger(__name__)

# --- [配置区] ---
# 文件: selection_manager/management/commands/prepare_stock_features.py
# --- [配置区] ---
LABEL_CONFIG = {
    # 'mode' 可选:
    #   'return': 未来N日收益率 (tanh缩放)
    #   'sharpe': 未来N日夏普比率 (收益和波动都向后看)
    #   'risk_adjusted_return': 风险调整后收益 (收益向后看，波动向前看)
    'mode': 'risk_adjusted_return',  # <--- 您可以根据需要切换这里
    
    'lookforward_days': 20,    # N值, 用于未来收益
    'lookback_days_vol': 20,   # 用于过去波动的回看窗口, 仅 'risk_adjusted_return' 模式使用
    
    'risk_free_rate_annual': 0.02, # 年化无风险利率, 仅 'sharpe' 模式使用
    'tanh_scaling_factor': 4.0,    # tanh缩放因子, 用于所有模式
}

FACTOR_LOOKBACK_BUFFER = 250 # 因子计算所需的最大回溯期, 넉넉하게
BATCH_SIZE = 100 # 每次处理100只股票

# --- [新增] 因子中文描述 (单一信息源) ---
FACTOR_DESCRIPTIONS = {
    # --- 原有因子 ---
    'dynamic_ADX_CONFIRM': '趋势动能-ADX趋势确认',
    'dynamic_v2_MA_SLOPE': '趋势动能-20日均线斜率(EMA平滑)',
    'dynamic_v2_MA_SCORE': '趋势动能-多均线顺排得分',
    'dynamic_v2_CPC_Factor': '趋势动能-收盘价位置因子',
    'dynamic_v2_VPCF': '趋势动能-量价相关因子',
    'dynamic_BREAKOUT_PWR': '强势突破-60日突破力度(ATR归一化)',
    'dynamic_VOLUME_SURGE': '强势突破-成交量激增',
    'dynamic_MOM_ACCEL': '强势突破-动量加速度',
    'dynamic_RSI_OS': '均值回归-RSI超卖指标',
    'dynamic_NEG_DEV': '均值回归-负向偏离度',
    'dynamic_BOLL_LB': '均值回归-布林带下轨位置',
    'dynamic_LOW_VOL': '质量防御-低波动率',
    'dynamic_MAX_DD': '质量防御-最大回撤',
    'dynamic_DOWNSIDE_RISK': '质量防御-下行风险',
    'dynamic_Old_D': '旧M值体系-方向函数D',
    'dynamic_Old_I': '旧M值体系-强度函数I',
    'dynamic_Old_M': '旧M值体系-综合M值',
    # --- 新增因子 ---
    'market_m_value': '市场状态-当日M值', # 注意：这个是直接使用的，不是lag1
    'market_m_value_lag1': '市场状态-昨日M值',
    'm_value_slope_5d': '市场趋势-M值5日回归斜率',
    'm_value_slope_20d': '市场趋势-M值20日回归斜率',
    'corporate_action_factor': '事件驱动-10日内除权息冲击',
    'turnover_slope_3d': '量能趋势-成交额3日回归斜率',
    'turnover_slope_10d': '量能趋势-成交额10日回归斜率',
    'avg_turnover_5d': '量能水平-5日平均成交额',
}


class Command(BaseCommand):
    help = '为个股评分模型生成特征和标签数据集 (X, y)。采用分批处理以优化内存。'

    def add_arguments(self, parser):
        parser.add_argument(
            '--use-local-db',
            action='store_true',
            help='如果指定，则从本地SQLite数据库读取DailyQuotes表。'
        )

    MODELS_DIR = settings.BASE_DIR / 'selection_manager' / 'ml_models'
    DATASET_FILE = MODELS_DIR / 'stock_features_dataset.pkl'
    MODEL_CONFIG_FILE = MODELS_DIR / 'stock_model_config.json'
    TEMP_DATA_DIR = MODELS_DIR / 'temp_feature_batches'

    def handle(self, *args, **options):
        self.epsilon=1e-9
        use_local_db = options['use_local_db']
        self.db_alias_quotes = 'local_sqlite' if use_local_db else 'default'
        self.db_alias_other = 'default'
        
        db_source_message = f"DailyQuotes from '{self.db_alias_quotes}', others from '{self.db_alias_other}'"
        self.stdout.write(self.style.SUCCESS(f"数据源配置: {db_source_message}"))
        self.stdout.write(self.style.SUCCESS("===== 开始为个股评分模型准备机器学习数据集 (分批处理模式) ====="))

        if self.TEMP_DATA_DIR.exists():
            shutil.rmtree(self.TEMP_DATA_DIR)
        self.TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.stdout.write(f"创建临时目录: {self.TEMP_DATA_DIR}")

        try:
            self.stdout.write("步骤 1/5: 加载全局数据 (股票列表, M值, 公司行动)...")
            global_data = self._load_global_data()
            if not global_data['all_stock_codes']:
                self.stdout.write(self.style.ERROR("错误: 数据库中没有可用的股票代码。"))
                return
            
            feature_names = self._get_feature_names()

            self.stdout.write(f"步骤 2/5: 开始分批处理 {len(global_data['all_stock_codes'])} 只股票，每批 {BATCH_SIZE} 只...")
            stock_batches = [global_data['all_stock_codes'][i:i + BATCH_SIZE] for i in range(0, len(global_data['all_stock_codes']), BATCH_SIZE)]
            batch_iterator = tqdm(enumerate(stock_batches), total=len(stock_batches), desc="处理批次")
            
            intermediate_files = []

            for i, batch_codes in batch_iterator:
                batch_iterator.set_description(f"处理批次 {i+1}/{len(stock_batches)}")
                
                quotes_df = self._load_and_filter_batch_quotes(batch_codes, global_data['listing_dates_df'])
                if quotes_df.empty:
                    self.stdout.write(self.style.WARNING(f"  - [批次 {i+1}] 筛选后无有效数据，已跳过。"))
                    continue
                
                labels_df = self._generate_labels(quotes_df)
                
                panel_data = self._build_panels(quotes_df)
                panel_data['m_value_series'] = global_data['m_values_series']
                panel_data['corp_action_panel'] = self._build_corp_action_panel(global_data['corp_actions_df'], panel_data['close'])

                calculator = StockFactorCalculator(panel_data, feature_names)
                features_df = calculator.run()
                
                # 将当日M值作为一个特征加入
                features_df['market_m_value'] = features_df.index.get_level_values('trade_date').map(global_data['m_values_series'])
                
                batch_final_df = features_df.join(labels_df, how='inner')
                batch_final_df.dropna(inplace=True)

                if batch_final_df.empty:
                    continue

                temp_file_path = self.TEMP_DATA_DIR / f'batch_{i}.pkl'
                with open(temp_file_path, 'wb') as f:
                    pickle.dump(batch_final_df, f)
                intermediate_files.append(temp_file_path)

            if not intermediate_files:
                self.stdout.write(self.style.ERROR("错误: 所有批次处理后未生成任何有效数据。"))
                return

            self.stdout.write(f"\n步骤 3/5 & 4/5: 合并 {len(intermediate_files)} 个中间文件...")
            all_dfs = [pd.read_pickle(file) for file in tqdm(intermediate_files, desc="合并文件")]
            final_df = pd.concat(all_dfs)

            self.stdout.write("步骤 5/5: 保存最终数据集和模型配置文件...")
            
            # 确保 'market_m_value' 在特征名列表中
            if 'market_m_value' not in feature_names:
                feature_names.append('market_m_value')

            X = final_df[feature_names]
            y = final_df['label']

            self.stdout.write(f"数据集准备完成。总样本数: {len(X)}")
            self.stdout.write("标签 (label) 统计信息:\n" + str(y.describe()))

            dataset = {'X': X, 'y': y, 'index': X.index, 'feature_names': feature_names}
            with open(self.DATASET_FILE, 'wb') as f:
                pickle.dump(dataset, f)
            self.stdout.write(self.style.SUCCESS(f"数据集已成功保存至: {self.DATASET_FILE}"))

            model_config = {'feature_names': feature_names}
            with open(self.MODEL_CONFIG_FILE, 'w') as f:
                json.dump(model_config, f, indent=4)
            self.stdout.write(self.style.SUCCESS(f"模型配置文件已成功保存至: {self.MODEL_CONFIG_FILE}"))
            
        finally:
            if self.TEMP_DATA_DIR.exists():
                shutil.rmtree(self.TEMP_DATA_DIR)
                self.stdout.write(self.style.SUCCESS(f"临时目录已清理: {self.TEMP_DATA_DIR}"))
        
        self.stdout.write(self.style.SUCCESS("===== 数据准备流程结束 ====="))

    def _load_global_data(self):
        """一次性加载全局共享数据"""
        # 1. 获取所有股票代码
        all_stock_codes = list(StockInfo.objects.using(self.db_alias_other).values_list('stock_code', flat=True).distinct())
        
        # 2. 获取M值时间序列
        m_values_qs = DailyFactorValues.objects.using(self.db_alias_other).filter(
            stock_code_id=MARKET_INDICATOR_CODE,
            factor_code_id='dynamic_M_VALUE'
        ).values('trade_date', 'raw_value')
        m_values_df = pd.DataFrame.from_records(m_values_qs)
        m_values_series = pd.Series(dtype=float)
        if not m_values_df.empty:
            m_values_series = m_values_df.set_index('trade_date')['raw_value'].astype(float)
            
        # 3. 获取所有公司行动数据
        corp_actions_qs = CorporateAction.objects.using(self.db_alias_other).filter(
            event_type__in=['dividend', 'bonus', 'transfer']
        ).values()
        corp_actions_df = pd.DataFrame.from_records(corp_actions_qs)

        # 4. 获取所有股票的首次交易日期
        listing_dates_qs = DailyQuotes.objects.using(self.db_alias_quotes) \
                                              .values('stock_code_id') \
                                              .annotate(first_trade_date=Min('trade_date'))
        listing_dates_df = pd.DataFrame.from_records(listing_dates_qs)
        listing_dates_df.rename(columns={'stock_code_id': 'stock_code'}, inplace=True)

        return {
            'all_stock_codes': all_stock_codes,
            'm_values_series': m_values_series,
            'corp_actions_df': corp_actions_df,
            'listing_dates_df': listing_dates_df
        }

    def _load_and_filter_batch_quotes(self, batch_codes: list, listing_dates_df: pd.DataFrame):
        """加载一个批次的行情数据并进行所有筛选"""
        # 1. 加载数据
        quotes_qs = DailyQuotes.objects.using(self.db_alias_quotes).filter(
            stock_code_id__in=batch_codes
        ).values('trade_date', 'stock_code_id', 'turnover', 'hfq_close')
        if not quotes_qs.exists():
            return pd.DataFrame()
        quotes_df = pd.DataFrame.from_records(quotes_qs)
        quotes_df.rename(columns={'stock_code_id': 'stock_code'}, inplace=True)

        # 2. 筛选ST股
        st_stocks = StockInfo.objects.using(self.db_alias_other).filter(
            stock_name__startswith='ST'
        ).values_list('stock_code', flat=True)
        quotes_df = quotes_df[~quotes_df['stock_code'].isin(st_stocks)]

        # 3. 筛选次新股
        quotes_df['trade_date'] = pd.to_datetime(quotes_df['trade_date']).dt.date
        quotes_df = pd.merge(quotes_df, listing_dates_df, on='stock_code', how='left')
        quotes_df.dropna(subset=['first_trade_date'], inplace=True)
        quotes_df['days_since_listing'] = (quotes_df['trade_date'] - quotes_df['first_trade_date']).apply(lambda x: x.days)
        quotes_df = quotes_df[quotes_df['days_since_listing'] >= 180]

        # 4. 筛选低成交额
        quotes_df['turnover'] = pd.to_numeric(quotes_df['turnover'], errors='coerce')
        quotes_df['avg_turnover_20d'] = quotes_df.groupby('stock_code')['turnover'].transform(
            lambda x: x.rolling(window=20, min_periods=15).mean()
        )
        quotes_df = quotes_df[quotes_df['avg_turnover_20d'] >= 100_000_000]
        
        # 5. 加载完整数据并复权
        if quotes_df.empty:
            return pd.DataFrame()
        
        # 获取筛选后剩余的 (date, code) 对
        filtered_keys = quotes_df[['trade_date', 'stock_code']].to_records(index=False)
        
        # 重新加载完整行情数据，但只加载需要的
        # 这是一个优化，但实现复杂，暂时先加载批次内全部数据再筛选
        full_quotes_qs = DailyQuotes.objects.using(self.db_alias_quotes).filter(
            stock_code_id__in=batch_codes
        ).values()
        full_quotes_df = pd.DataFrame.from_records(full_quotes_qs)
        full_quotes_df.rename(columns={'stock_code_id': 'stock_code'}, inplace=True)
        
        # 合并以应用筛选
        final_df = pd.merge(full_quotes_df, quotes_df[['trade_date', 'stock_code']], on=['trade_date', 'stock_code'], how='inner')

        # 复权
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'hfq_close']
        for col in numeric_cols:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        final_df['adj_factor'] = final_df['hfq_close'] / (final_df['close'] + 1e-9)
        final_df['open'] = final_df['open'] * final_df['adj_factor']
        final_df['high'] = final_df['high'] * final_df['adj_factor']
        final_df['low'] = final_df['low'] * final_df['adj_factor']
        final_df['close'] = final_df['hfq_close']
        final_df.rename(columns={'turnover': 'amount'}, inplace=True)
        
        return final_df[['trade_date', 'stock_code', 'open', 'high', 'low', 'close', 'volume', 'amount']]

    def _generate_labels(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        df = quotes_df.set_index(['trade_date', 'stock_code'])['close'].unstack()
        df.replace(0, np.nan, inplace=True)
        
        N_forward = LABEL_CONFIG['lookforward_days']
        
        # --- 计算每日收益率 (sharpe 和 risk_adjusted_return 模式都需要) ---
        daily_returns = df.pct_change(fill_method=None)
        # --- 根据模式选择不同的标签计算逻辑 ---
        mode = LABEL_CONFIG['mode']
        
        if mode == 'return':
            # 模式1: 未来N日收益率
            self.stdout.write(f"  - [标签模式: return] 计算未来 {N_forward} 日收益率...")
            future_price = df.shift(-N_forward)
            labels = (future_price / df) - 1
        elif mode == 'sharpe':
            # 模式2: 未来N日夏普比率 (收益和波动都向后看)
            self.stdout.write(f"  - [标签模式: sharpe] 计算未来 {N_forward} 日夏普比率...")
            daily_rf = (1 + LABEL_CONFIG['risk_free_rate_annual'])**(1/252) - 1
            excess_returns = daily_returns - daily_rf
            
            # 计算未来N日的滚动均值和标准差
            future_mean = excess_returns.shift(-N_forward).rolling(window=N_forward).mean()
            future_std = excess_returns.shift(-N_forward).rolling(window=N_forward).std()
            
            # 计算年化夏普比率
            annualized_sharpe = (future_mean / (future_std + self.epsilon)) * np.sqrt(252)
            labels = annualized_sharpe
        elif mode == 'risk_adjusted_return':
            # 模式3: 风险调整后收益 (收益向后看，波动向前看)
            self.stdout.write(f"  - [标签模式: risk_adjusted_return] 计算风险调整后收益...")
            N_lookback_vol = LABEL_CONFIG['lookback_days_vol']
            
            # 收益向后看N天
            future_price = df.shift(-N_forward)
            forward_return = (future_price / df) - 1
            
            # 波动向前看N天 (即历史N日波动率)
            past_volatility = (daily_returns.rolling(window=N_lookback_vol).std())* np.sqrt(N_forward)
            
            # 计算标签
            labels = forward_return / (past_volatility + self.epsilon)
        else:
            raise ValueError(f"未知的标签模式: {mode}")
        # 对所有模式的最终结果应用tanh进行缩放
        scaled_labels = np.tanh(LABEL_CONFIG['tanh_scaling_factor'] * labels)
        
        return scaled_labels.stack().rename('label').to_frame()

    def _build_panels(self, quotes_df: pd.DataFrame) -> dict:
        """将长格式的DataFrame转换为面板字典"""
        panel_data = {}
        quotes_df['trade_date'] = pd.to_datetime(quotes_df['trade_date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            panel = quotes_df.pivot(index='trade_date', columns='stock_code', values=col)
            if not panel.empty:
                full_date_range = pd.date_range(start=panel.index.min(), end=panel.index.max(), freq='B')
                panel = panel.reindex(full_date_range).ffill()
            panel_data[col] = panel
        return panel_data

    def _build_corp_action_panel(self, corp_actions_df: pd.DataFrame, close_panel: pd.DataFrame) -> pd.DataFrame:
        """构建公司行动因子面板"""
        if corp_actions_df.empty:
            return pd.DataFrame(0, index=close_panel.index, columns=close_panel.columns)

        corp_actions_df['ex_dividend_date'] = pd.to_datetime(corp_actions_df['ex_dividend_date'])
        
        # 创建一个空的因子面板
        action_factor_panel = pd.DataFrame(0.0, index=close_panel.index, columns=close_panel.columns)

        # 按股票和日期分组事件
        events_grouped = corp_actions_df.groupby(['stock_code', 'ex_dividend_date'])

        for (stock_code, date), group in events_grouped:
            if stock_code not in close_panel.columns or date not in close_panel.index:
                continue
            
            prev_date = date - pd.Timedelta(days=1)
            if prev_date not in close_panel.index:
                continue
            
            prev_close = close_panel.loc[prev_date, stock_code]
            if pd.isna(prev_close) or prev_close <= 0:
                continue

            # 计算综合调整率
            price_ratio = 1.0
            # 先处理送转股
            for _, event in group.iterrows():
                if event['event_type'] in ['bonus', 'transfer'] and event['shares_before'] and event['shares_after']:
                    price_ratio *= float(event['shares_before']) / float(event['shares_after'])
            
            # 用调整后的价格计算分红影响
            adjusted_prev_close = prev_close * price_ratio
            
            # 再处理分红
            for _, event in group.iterrows():
                if event['event_type'] == 'dividend' and event['dividend_per_share']:
                    price_ratio *= (adjusted_prev_close - float(event['dividend_per_share'])) / adjusted_prev_close

            if price_ratio > 0:
                factor_value = (1 / price_ratio) - 1
                action_factor_panel.loc[date, stock_code] = factor_value
        
        return action_factor_panel

    def _get_feature_names(self):
        """从M值模型配置中获取基础因子列表，并追加新因子"""
        try:
            with open(self.MODELS_DIR / 'm_value_model_config.json', 'r') as f:
                m_value_config = json.load(f)
            base_features = m_value_config['feature_names']
            self.stdout.write("成功从'm_value_model_config.json'加载基础因子列表。")
        except FileNotFoundError:
            self.stdout.write(self.style.WARNING("M值模型配置文件不存在，将使用默认的基础因子列表。"))
            base_features = [
                'dynamic_ADX_CONFIRM', 'dynamic_v2_MA_SLOPE', 'dynamic_v2_MA_SCORE',
                'dynamic_v2_CPC_Factor', 'dynamic_v2_VPCF', 'dynamic_BREAKOUT_PWR',
                'dynamic_VOLUME_SURGE', 'dynamic_MOM_ACCEL', 'dynamic_RSI_OS',
                'dynamic_NEG_DEV', 'dynamic_BOLL_LB', 'dynamic_LOW_VOL',
                'dynamic_MAX_DD', 'dynamic_DOWNSIDE_RISK', 'dynamic_Old_D',
                'dynamic_Old_I', 'dynamic_Old_M'
            ]
        
        # 追加新因子
        new_features = [
            'market_m_value_lag1',
            'm_value_slope_5d',
            'm_value_slope_20d',
            'corporate_action_factor',
            'turnover_slope_3d',
            'turnover_slope_10d',
            'avg_turnover_5d',
        ]
        
        return base_features + new_features
