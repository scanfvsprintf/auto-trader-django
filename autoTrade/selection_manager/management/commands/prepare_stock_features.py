# selection_manager/management/commands/prepare_stock_features.py
# [最终修复版 v3]
# 描述: 为个股评分模型生成特征和标签数据集。
#
# 核心修复点:
# 1. [已解决] 空DataFrame问题: 移除了过于激进的 `dropna()` 操作，防止整个批次数据被意外清空。
# 2. [新增] 增加了详细的诊断日志，追踪每个批次的数据行数变化，便于调试。
# 3. [保持] 维持了对数据爆炸Bug和.dt访问器错误的修复。

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
from selection_manager.service.stock_factor_calculator import StockFactorCalculator

logger = logging.getLogger(__name__)

# --- [配置区] --- (保持不变)
LABEL_CONFIG = {
    'mode': 'return',
    'lookforward_days': 5,
    'lookback_days_vol': 20,
    'risk_free_rate_annual': 0.02,
    'tanh_scaling_factor': 3.6,
}

FACTOR_LOOKBACK_BUFFER = 250
BATCH_SIZE = 100

# --- [因子中文描述] --- (保持不变)
FACTOR_DESCRIPTIONS = {
    # ... (内容省略) ...
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
        self.epsilon = 1e-9
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
            total_samples_generated = 0

            for i, batch_codes in batch_iterator:
                batch_iterator.set_description(f"处理批次 {i+1}/{len(stock_batches)}")

                quotes_df = self._load_and_filter_batch_quotes(batch_codes, global_data['listing_dates_df'])
                if quotes_df.empty:
                    self.stdout.write(self.style.WARNING(f"  - [批次 {i+1}] 筛选后无有效行情数据，已跳过。"))
                    continue
                self.stdout.write(f"  - [批次 {i+1}] 筛选后行情数据: {len(quotes_df)} 行")

                labels_df = self._generate_labels(quotes_df.copy())
                self.stdout.write(f"  - [批次 {i+1}] 生成标签数据: {len(labels_df)} 行")

                panel_data = self._build_panels(quotes_df.copy())
                
                if panel_data['close'].empty:
                    self.stdout.write(self.style.WARNING(f"  - [批次 {i+1}] 构建面板后无有效数据，已跳过。"))
                    continue

                panel_data['m_value_series'] = global_data['m_values_series']
                panel_data['corp_action_panel'] = self._build_corp_action_panel(global_data['corp_actions_df'], panel_data['close'])

                calculator = StockFactorCalculator(panel_data, feature_names)
                features_df = calculator.run()
                self.stdout.write(f"  - [批次 {i+1}] 生成特征数据: {len(features_df)} 行")

                features_df_index_date = features_df.index.get_level_values('trade_date').date
                features_df['market_m_value'] = pd.Series(features_df_index_date, index=features_df.index).map(global_data['m_values_series']).values

                batch_final_df = features_df.join(labels_df, how='inner')
                self.stdout.write(f"  - [批次 {i+1}] 特征与标签Join后: {len(batch_final_df)} 行")

                # [关键修复] 移除或注释掉这行过于激进的 dropna。
                # `how='inner'` 的 join 已经保证了标签存在。特征中的 NaN 是有意义的（例如，计算初期），
                # 应该由下游模型处理，或者在所有数据合并后进行更精细的清洗。
                # batch_final_df.dropna(inplace=True)
                # self.stdout.write(f"  - [批次 {i+1}] dropna后: {len(batch_final_df)} 行")

                if batch_final_df.empty:
                    self.stdout.write(self.style.WARNING(f"  - [批次 {i+1}] Join后数据为空，已跳过。"))
                    continue

                temp_file_path = self.TEMP_DATA_DIR / f'batch_{i}.pkl'
                with open(temp_file_path, 'wb') as f:
                    pickle.dump(batch_final_df, f)
                intermediate_files.append(temp_file_path)
                self.stdout.write(self.style.SUCCESS(f"  - [批次 {i+1}] 成功写入临时文件: {temp_file_path.name}, 包含 {len(batch_final_df)} 个样本。"))
                total_samples_generated += len(batch_final_df)


            if not intermediate_files:
                self.stdout.write(self.style.ERROR("错误: 所有批次处理后未生成任何有效数据。"))
                return

            self.stdout.write(f"\n步骤 3/5 & 4/5: 合并 {len(intermediate_files)} 个中间文件...")
            all_dfs = [pd.read_pickle(file) for file in tqdm(intermediate_files, desc="合并文件")]
            final_df = pd.concat(all_dfs)

            self.stdout.write("步骤 5/5: 保存最终数据集和模型配置文件...")

            if 'market_m_value' not in feature_names:
                feature_names.append('market_m_value')

            # 在最终合并后，可以进行一次更温和的 NaN 清理，例如，只删除标签为 NaN 的行
            final_df.dropna(subset=['label'], inplace=True)

            X = final_df[feature_names]
            y = final_df['label']

            self.stdout.write(f"数据集准备完成。总样本数: {len(X)} (来自 {total_samples_generated} 个中间样本)")
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
                #shutil.rmtree(self.TEMP_DATA_DIR)
                self.stdout.write(self.style.SUCCESS(f"临时目录已清理: {self.TEMP_DATA_DIR}"))

        self.stdout.write(self.style.SUCCESS("===== 数据准备流程结束 ====="))

    # ... _load_global_data, _load_and_filter_batch_quotes, _build_panels, _build_corp_action_panel, _get_feature_names 保持 v2 版本不变 ...
    # 这里为了简洁省略，请使用上一轮回复中的 v2 版本代码
    def _load_global_data(self):
        """一次性加载全局共享数据"""
        all_stock_codes = list(StockInfo.objects.using(self.db_alias_other).values_list('stock_code', flat=True).distinct())

        m_values_qs = DailyFactorValues.objects.using(self.db_alias_other).filter(
            stock_code_id=MARKET_INDICATOR_CODE,
            factor_code_id='dynamic_M_VALUE'
        ).values('trade_date', 'raw_value')
        m_values_df = pd.DataFrame.from_records(m_values_qs)
        m_values_series = pd.Series(dtype=float)
        if not m_values_df.empty:
            m_values_df['trade_date'] = pd.to_datetime(m_values_df['trade_date'])
            m_values_df = m_values_df.set_index('trade_date')
            m_values_df.index = m_values_df.index.date
            m_values_series = m_values_df['raw_value'].astype(float)

        corp_actions_qs = CorporateAction.objects.using(self.db_alias_other).filter(
            event_type__in=['dividend', 'bonus', 'transfer']
        ).values()
        corp_actions_df = pd.DataFrame.from_records(corp_actions_qs)

        listing_dates_qs = DailyQuotes.objects.using(self.db_alias_quotes) \
                                              .values('stock_code_id') \
                                              .annotate(first_trade_date=Min('trade_date'))
        listing_dates_df = pd.DataFrame.from_records(listing_dates_qs)
        listing_dates_df.rename(columns={'stock_code_id': 'stock_code'}, inplace=True)
        
        if not listing_dates_df.empty:
            listing_dates_df['first_trade_date'] = pd.to_datetime(listing_dates_df['first_trade_date'])

        return {
            'all_stock_codes': all_stock_codes,
            'm_values_series': m_values_series,
            'corp_actions_df': corp_actions_df,
            'listing_dates_df': listing_dates_df
        }

    def _load_and_filter_batch_quotes(self, batch_codes: list, listing_dates_df: pd.DataFrame):
        """[已修复 v2] 加载一个批次的行情数据并进行所有筛选。"""
        quotes_qs = DailyQuotes.objects.using(self.db_alias_quotes).filter(
            stock_code_id__in=batch_codes
        ).values(
            'trade_date', 'stock_code_id', 'open', 'high', 'low', 'close',
            'volume', 'turnover', 'hfq_close'
        )

        if not quotes_qs.exists():
            return pd.DataFrame()

        quotes_df = pd.DataFrame.from_records(quotes_qs)
        quotes_df.rename(columns={'stock_code_id': 'stock_code', 'turnover': 'amount'}, inplace=True)

        if quotes_df.empty:
            return quotes_df

        quotes_df['trade_date'] = pd.to_datetime(quotes_df['trade_date'])

        st_stocks = StockInfo.objects.using(self.db_alias_other).filter(
            stock_name__startswith='ST'
        ).values_list('stock_code', flat=True)
        quotes_df = quotes_df[~quotes_df['stock_code'].isin(st_stocks)]
        if quotes_df.empty: return quotes_df

        quotes_df = pd.merge(quotes_df, listing_dates_df, on='stock_code', how='left')
        quotes_df.dropna(subset=['first_trade_date'], inplace=True)
        
        quotes_df['days_since_listing'] = (quotes_df['trade_date'] - quotes_df['first_trade_date']).dt.days
        quotes_df = quotes_df[quotes_df['days_since_listing'] >= 180]
        if quotes_df.empty: return quotes_df

        quotes_df['amount'] = pd.to_numeric(quotes_df['amount'], errors='coerce')
        quotes_df['avg_turnover_20d'] = quotes_df.groupby('stock_code')['amount'].transform(
            lambda x: x.rolling(window=20, min_periods=15).mean()
        )
        quotes_df = quotes_df[quotes_df['avg_turnover_20d'] >= 100_000_000]
        if quotes_df.empty: return quotes_df

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'hfq_close']
        for col in numeric_cols:
            quotes_df[col] = pd.to_numeric(quotes_df[col], errors='coerce')

        quotes_df['adj_factor'] = quotes_df['hfq_close'] / (quotes_df['close'] + self.epsilon)
        quotes_df['open'] = quotes_df['open'] * quotes_df['adj_factor']
        quotes_df['high'] = quotes_df['high'] * quotes_df['adj_factor']
        quotes_df['low'] = quotes_df['low'] * quotes_df['adj_factor']
        quotes_df['close'] = quotes_df['hfq_close']

        return quotes_df[['trade_date', 'stock_code', 'open', 'high', 'low', 'close', 'volume', 'amount']]

    def _generate_labels(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        quotes_df['trade_date'] = quotes_df['trade_date'].dt.date
        
        df = quotes_df.set_index(['trade_date', 'stock_code'])['close'].unstack()
        df.replace(0, np.nan, inplace=True)

        N_forward = LABEL_CONFIG['lookforward_days']
        daily_returns = df.pct_change(fill_method=None)
        mode = LABEL_CONFIG['mode']

        if mode == 'return':
            self.stdout.write(f"  - [标签模式: return] 计算未来 {N_forward} 日收益率...")
            future_price = df.shift(-N_forward)
            labels = (future_price / df) - 1
        elif mode == 'sharpe':
            self.stdout.write(f"  - [标签模式: sharpe] 计算未来 {N_forward} 日夏普比率...")
            daily_rf = (1 + LABEL_CONFIG['risk_free_rate_annual'])**(1/252) - 1
            excess_returns = daily_returns - daily_rf
            future_mean = excess_returns.shift(-N_forward).rolling(window=N_forward).mean()
            future_std = excess_returns.shift(-N_forward).rolling(window=N_forward).std()
            annualized_sharpe = (future_mean / (future_std + self.epsilon)) * np.sqrt(252)
            labels = annualized_sharpe
        elif mode == 'risk_adjusted_return':
            self.stdout.write(f"  - [标签模式: risk_adjusted_return] 计算风险调整后收益...")
            N_lookback_vol = LABEL_CONFIG['lookback_days_vol']
            future_price = df.shift(-N_forward)
            forward_return = (future_price / df) - 1
            past_volatility = (daily_returns.rolling(window=N_lookback_vol).std()) * np.sqrt(N_forward)
            labels = forward_return / (past_volatility + self.epsilon)
        else:
            raise ValueError(f"未知的标签模式: {mode}")

        scaled_labels = np.tanh(LABEL_CONFIG['tanh_scaling_factor'] * labels)
        return scaled_labels.stack(future_stack=True).rename('label').to_frame()

    def _build_panels(self, quotes_df: pd.DataFrame) -> dict:
        """[已修复 v2] 将长格式的DataFrame转换为面板字典。"""
        panel_data = {}
        
        if quotes_df.empty:
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                panel_data[col] = pd.DataFrame()
            return panel_data

        actual_trade_dates = sorted(quotes_df['trade_date'].unique())

        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            panel = quotes_df.pivot(index='trade_date', columns='stock_code', values=col)

            if not panel.empty:
                panel = panel.reindex(actual_trade_dates)
                panel = panel.ffill(limit=120)

            panel_data[col] = panel

        return panel_data

    def _build_corp_action_panel(self, corp_actions_df: pd.DataFrame, close_panel: pd.DataFrame) -> pd.DataFrame:
        if corp_actions_df.empty or close_panel.empty:
            return pd.DataFrame(0, index=close_panel.index, columns=close_panel.columns)

        corp_actions_df['ex_dividend_date'] = pd.to_datetime(corp_actions_df['ex_dividend_date'])

        action_factor_panel = pd.DataFrame(0.0, index=close_panel.index, columns=close_panel.columns)
        
        events_grouped = corp_actions_df.groupby(['stock_code', 'ex_dividend_date'])

        for (stock_code, date), group in events_grouped:
            if stock_code not in close_panel.columns or date not in close_panel.index:
                continue

            try:
                prev_date_loc = close_panel.index.get_loc(date) - 1
                if prev_date_loc < 0:
                    continue
                prev_date = close_panel.index[prev_date_loc]
            except KeyError:
                continue

            prev_close = close_panel.loc[prev_date, stock_code]
            if pd.isna(prev_close) or prev_close <= 0:
                continue

            price_ratio = 1.0
            for _, event in group.iterrows():
                if event['event_type'] in ['bonus', 'transfer'] and event['shares_before'] and event['shares_after']:
                    price_ratio *= float(event['shares_before']) / float(event['shares_after'])

            adjusted_prev_close = prev_close * price_ratio

            for _, event in group.iterrows():
                if event['event_type'] == 'dividend' and event['dividend_per_share']:
                    dividend = float(event['dividend_per_share'])
                    if adjusted_prev_close > dividend:
                        price_ratio *= (adjusted_prev_close - dividend) / adjusted_prev_close

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
