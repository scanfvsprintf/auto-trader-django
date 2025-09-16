# -*- coding: utf-8 -*-
"""
基于akshare的ETF策略回测脚本 (V9 - 增加高级统计分析与数据缓存)

V9更新:
- 增加数据缓存功能，避免重复从网络获取数据。
- 增加统计图1: 移动最大回撤比率。
- 增加统计图2: 持有期分析（最大/最小收益率，胜率）。
- 增加统计图3: 两只ETF的扩张窗口协方差。
- 每个统计图的数据都保存为独立的CSV文件。
- 将所有分析和绘图功能模块化，保持代码整洁。
"""

import akshare as ak
import pandas as pd
import math
import time
import logging
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter

# ==============================================================================
# 模块0: 日志设置
# ==============================================================================
def setup_logging():
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('backtest_log.txt', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# ==============================================================================
# 模块1: 配置
# ==============================================================================
CONFIG = {
    "start_date": "2016-02-17",
    "end_date": "2025-09-08",
    "initial_capital": 200000.0,
    "etf_codes": {
        "nasdaq": "513100",
        "dividend": "510880"
    },
    "target_allocation": {
        "nasdaq": 0.45,
        "dividend": 0.45,
        "cash": 0.10
    },
    "rebalance_threshold": 0.10,
    "cash_buffer": {
        "min_ratio": 0.05,
        "max_ratio": 0.15,
        "target_ratio": 0.10
    },
    "monthly_living_expense": 0.0,
    "commission_rate": 0.0003,
    "min_commission": 5.0,
    
    # --- 新增配置 ---
    "data_cache_path": "etf_data_cache.csv",
    "output_dir": "backtest_outputs", # 用于存放所有输出文件的目录
    "max_holding_days": 252 * 3, # 持有期分析的最大天数 (约3年)
}

# ==============================================================================
# 模块2: 数据准备 (Data Preparation) - V9版逻辑 (增加缓存)
# ==============================================================================
def get_prepared_data(config: dict) -> pd.DataFrame:
    logging.info("开始获取和准备数据...")
    
    cache_path = config["data_cache_path"]
    start_date_req = pd.to_datetime(config["start_date"])
    end_date_req = pd.to_datetime(config["end_date"])

    # 检查缓存文件
    if os.path.exists(cache_path):
        logging.info(f"发现缓存文件: {cache_path}")
        try:
            cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cache_start = cached_df.index.min()
            cache_end = cached_df.index.max()
            
            # 检查缓存日期范围是否覆盖需求范围
            if cache_start <= start_date_req and cache_end >= end_date_req:
                logging.info("缓存数据范围满足需求，将使用缓存数据。")
                # 截取所需日期范围并返回
                return cached_df.loc[start_date_req:end_date_req].copy()
            else:
                logging.warning("缓存数据范围不足，将重新从网络获取全部数据。")
                os.remove(cache_path) # 删除旧缓存
        except Exception as e:
            logging.error(f"读取缓存文件失败: {e}。将重新从网络获取数据。")
            if os.path.exists(cache_path):
                os.remove(cache_path)

    logging.info("无有效缓存，开始从akshare获取数据。")
    logging.info("采用方法：通过对比不复权和后复权价格，手动计算分红。")
    
    fetch_start_date = (start_date_req - pd.DateOffset(months=1)).strftime("%Y%m%d")
    fetch_end_date = end_date_req.strftime("%Y%m%d")

    all_etf_data = []

    for name, code in config["etf_codes"].items():
        try:
            logging.info(f"正在获取 {name} ({code}) 的不复权行情数据...")
            raw_df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=fetch_start_date, end_date=fetch_end_date, adjust="")
            raw_df['日期'] = pd.to_datetime(raw_df['日期'])
            raw_df.set_index('日期', inplace=True)
            raw_df = raw_df[['开盘', '收盘']].rename(columns={'开盘': f'open_{name}', '收盘': f'close_raw_{name}'})
            time.sleep(1)

            logging.info(f"正在获取 {name} ({code}) 的后复权行情数据...")
            hfq_df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=fetch_start_date, end_date=fetch_end_date, adjust="hfq")
            hfq_df['日期'] = pd.to_datetime(hfq_df['日期'])
            hfq_df.set_index('日期', inplace=True)
            hfq_df = hfq_df[['收盘']].rename(columns={'收盘': f'close_hfq_{name}'})
            time.sleep(1)

            merged_df = pd.concat([raw_df, hfq_df], axis=1)
            merged_df.sort_index(inplace=True)

            merged_df[f'close_raw_prev_{name}'] = merged_df[f'close_raw_{name}'].shift(1)
            merged_df[f'close_hfq_prev_{name}'] = merged_df[f'close_hfq_{name}'].shift(1)
            
            dividend = merged_df[f'close_raw_prev_{name}'] * \
                       (merged_df[f'close_hfq_{name}'] / merged_df[f'close_hfq_prev_{name}']) - \
                       merged_df[f'close_raw_{name}']
            
            dividend[dividend < 0.0001] = 0
            merged_df[f'dividend_{name}'] = dividend
            
            # 保留 open, dividend 和 hfq_close 用于后续分析
            final_etf_df = merged_df[[f'open_{name}', f'dividend_{name}', f'close_hfq_{name}']]
            all_etf_data.append(final_etf_df)
            
            detected_dividends = final_etf_df[final_etf_df[f'dividend_{name}'] > 0]
            if not detected_dividends.empty:
                logging.info(f"为 {name} ({code}) 检测到以下分红事件:")
                for date, row in detected_dividends.iterrows():
                    logging.info(f"  - 日期: {date.strftime('%Y-%m-%d')}, 每股分红(约): {row[f'dividend_{name}']:.4f}")
            else:
                logging.info(f"在指定时间范围内未检测到 {name} ({code}) 的分红事件。")

        except Exception as e:
            logging.error(f"处理 {name} ({code}) 数据时发生严重错误: {e}")
            return pd.DataFrame()

    logging.info("正在合并所有ETF的最终数据...")
    master_df = pd.concat(all_etf_data, axis=1)
    master_df.sort_index(inplace=True)
    
    # 在保存缓存前不截断日期，以保留完整获取的数据
    try:
        master_df.to_csv(cache_path)
        logging.info(f"数据已成功缓存至: {cache_path}")
    except Exception as e:
        logging.error(f"保存数据缓存失败: {e}")

    # 截取所需日期范围
    master_df = master_df.loc[start_date_req:end_date_req]
    
    open_cols = [f'open_{name}' for name in config["etf_codes"]]
    master_df.dropna(subset=open_cols, inplace=True)
    
    master_df.ffill(inplace=True)
    
    for name in config["etf_codes"].keys():
        if f'dividend_{name}' in master_df.columns:
            master_df[f'dividend_{name}'].fillna(0, inplace=True)
        else:
            master_df[f'dividend_{name}'] = 0.0

    if master_df.empty:
        logging.error("数据准备失败，无有效数据用于回测。")
        return pd.DataFrame()

    logging.info("数据准备完成！")
    logging.info(f"回测将在 {master_df.index.min().strftime('%Y-%m-%d')} 到 {master_df.index.max().strftime('%Y-%m-%d')} 之间进行。")
    
    return master_df

# ==============================================================================
# 模块3: 回测引擎 (Backtesting Engine) - (无需改动)
# ==============================================================================
def run_backtest(config: dict, data_df: pd.DataFrame):
    # (此函数实现与上一版本完全相同)
    if data_df.empty:
        logging.error("数据为空，无法开始回测。")
        return pd.DataFrame()

    def _calculate_commission(trade_value: float) -> float:
        commission = trade_value * config["commission_rate"]
        return max(commission, config["min_commission"])

    def _buy(portfolio: dict, symbol_name: str, shares_to_buy: int, price: float) -> bool:
        if shares_to_buy <= 0: return False
        cost = shares_to_buy * price
        commission = _calculate_commission(cost)
        total_cost = cost + commission
        if portfolio['cash'] < total_cost:
            logging.warning(f"现金不足 ({portfolio['cash']:.2f})，无法买入 {shares_to_buy} 股 {symbol_name} (需 {total_cost:.2f})。")
            return False
        portfolio['cash'] -= total_cost
        portfolio[f'{symbol_name}_shares'] += shares_to_buy
        logging.info(f"[买入] {shares_to_buy} 股 {symbol_name} @ {price:.2f}, 花费: {total_cost:.2f}")
        return True

    def _sell(portfolio: dict, symbol_name: str, shares_to_sell: int, price: float) -> float:
        if shares_to_sell <= 0: return 0.0
        if portfolio[f'{symbol_name}_shares'] < shares_to_sell:
            logging.warning(f"持仓不足 ({portfolio[f'{symbol_name}_shares']})，无法卖出 {shares_to_sell} 股 {symbol_name}。")
            return 0.0
        revenue = shares_to_sell * price
        commission = _calculate_commission(revenue)
        net_revenue = revenue - commission
        portfolio['cash'] += net_revenue
        portfolio[f'{symbol_name}_shares'] -= shares_to_sell
        logging.info(f"[卖出] {shares_to_sell} 股 {symbol_name} @ {price:.2f}, 净收入: {net_revenue:.2f}")
        return net_revenue

    def update_portfolio_values(portfolio: dict, prices: dict) -> dict:
        nasdaq_value = portfolio['nasdaq_shares'] * prices['nasdaq']
        dividend_value = portfolio['dividend_shares'] * prices['dividend']
        total_assets = portfolio['cash'] + nasdaq_value + dividend_value
        return {"nasdaq_value": nasdaq_value, "dividend_value": dividend_value, "total_assets": total_assets}

    portfolio = {'cash': config["initial_capital"], 'nasdaq_shares': 0, 'dividend_shares': 0}
    results = []
    last_month_processed = None
    is_first_day = True

    for date, row in data_df.iterrows():
        logging.info(f"--- 交易日: {date.strftime('%Y-%m-%d')} ---")

        prices = {name: row[f'open_{name}'] for name in config["etf_codes"].keys()}
        if any(p <= 0 for p in prices.values()):
            logging.warning("当日价格数据异常(<=0)，跳过所有操作。")
            if results:
                last_record = results[-1].copy(); last_record['date'] = date; results.append(last_record)
            continue

        current_values = update_portfolio_values(portfolio, prices)
        logging.info(f"盘前状态: 总资产 {current_values['total_assets']:,.2f}, 现金 {portfolio['cash']:,.2f}, "
                     f"纳指ETF {current_values['nasdaq_value']:,.2f}, 红利ETF {current_values['dividend_value']:,.2f}")

        if is_first_day:
            logging.info("执行首次建仓...")
            target_nasdaq_value = config["initial_capital"] * config["target_allocation"]["nasdaq"]
            target_dividend_value = config["initial_capital"] * config["target_allocation"]["dividend"]
            nasdaq_shares_to_buy = math.floor(target_nasdaq_value / prices['nasdaq'] / 100) * 100
            _buy(portfolio, 'nasdaq', nasdaq_shares_to_buy, prices['nasdaq'])
            dividend_shares_to_buy = math.floor(target_dividend_value / prices['dividend'] / 100) * 100
            _buy(portfolio, 'dividend', dividend_shares_to_buy, prices['dividend'])
            is_first_day = False
            current_values = update_portfolio_values(portfolio, prices)

        dividend_received_today = False
        for name in config["etf_codes"].keys():
            dividend_per_share = row[f'dividend_{name}']
            if dividend_per_share > 0 and portfolio[f'{name}_shares'] > 0:
                cash_from_dividend = portfolio[f'{name}_shares'] * dividend_per_share
                portfolio['cash'] += cash_from_dividend
                dividend_received_today = True
                logging.info(f"[分红] {name} ETF派息, 每股 {dividend_per_share:.4f}元, 共收到 {cash_from_dividend:,.2f}元现金。")
        if dividend_received_today:
            current_values = update_portfolio_values(portfolio, prices)

        current_month = date.month
        if last_month_processed != current_month:
            expense = config["monthly_living_expense"]
            if portfolio['cash'] >= expense:
                portfolio['cash'] -= expense
                logging.info(f"[支出] 扣除月度生活费 {expense:,.2f}元。")
                current_values = update_portfolio_values(portfolio, prices)
            else:
                logging.warning(f"现金不足，无法扣除生活费 {expense:,.2f}元。")
            last_month_processed = current_month

        if current_values["total_assets"] > 0:
            cash_ratio = portfolio['cash'] / current_values["total_assets"]
            logging.info(f"现金管理检查: 现金比例 {cash_ratio:.2%} (阈值: {config['cash_buffer']['min_ratio']:.0%}-{config['cash_buffer']['max_ratio']:.0%})")

            if cash_ratio < config["cash_buffer"]["min_ratio"]:
                logging.info("-> 触发: 现金比例过低。")
                cash_needed = (config["cash_buffer"]["target_ratio"] * current_values["total_assets"]) - portfolio['cash']
                if cash_needed > 0 and portfolio['dividend_shares'] > 0:
                    value_to_sell = cash_needed / (1 - config["commission_rate"])
                    shares_to_sell = min(math.ceil(value_to_sell / prices['dividend'] / 100) * 100, portfolio['dividend_shares'])
                    if _sell(portfolio, 'dividend', shares_to_sell, prices['dividend']) > 0:
                        current_values = update_portfolio_values(portfolio, prices)

            elif cash_ratio > config["cash_buffer"]["max_ratio"] and dividend_received_today:
                logging.info("-> 触发: 因分红导致现金比例过高。")
                excess_cash = portfolio['cash'] - (config["cash_buffer"]["max_ratio"] * current_values["total_assets"])
                if excess_cash > 0:
                    shares_to_buy = math.floor((excess_cash / (1 + config["commission_rate"])) / prices['dividend'] / 100) * 100
                    if _buy(portfolio, 'dividend', shares_to_buy, prices['dividend']):
                        current_values = update_portfolio_values(portfolio, prices)

        if current_values["total_assets"] > 0:
            nasdaq_weight = current_values["nasdaq_value"] / current_values["total_assets"]
            dividend_weight = current_values["dividend_value"] / current_values["total_assets"]
            logging.info(f"调仓检查: 纳指权重 {nasdaq_weight:.2%}, 红利权重 {dividend_weight:.2%}, "
                         f"权重差 {abs(nasdaq_weight - dividend_weight):.2%} (阈值: {config['rebalance_threshold']:.0%})")

            if abs(nasdaq_weight - dividend_weight) > config["rebalance_threshold"]:
                logging.info("-> 触发: 调仓。")
                etf_total_value = current_values["nasdaq_value"] + current_values["dividend_value"]
                target_etf_value = etf_total_value / 2.0

                if nasdaq_weight > dividend_weight:
                    logging.info("调仓方向: 卖出纳指ETF，买入红利ETF")
                    value_to_rebalance = current_values["nasdaq_value"] - target_etf_value
                    nasdaq_shares_to_sell = math.floor((value_to_rebalance / prices['nasdaq']) / 100) * 100
                    proceeds = _sell(portfolio, 'nasdaq', nasdaq_shares_to_sell, prices['nasdaq'])
                    if proceeds > 0:
                        shares_to_buy = math.floor((proceeds / (1 + config["commission_rate"])) / prices['dividend'] / 100) * 100
                        _buy(portfolio, 'dividend', shares_to_buy, prices['dividend'])
                else:
                    logging.info("调仓方向: 买入纳指ETF，资金来源: 1.现金 2.卖出红利ETF")
                    value_to_buy_nasdaq = target_etf_value - current_values["nasdaq_value"]
                    cash_min_level = config["cash_buffer"]["min_ratio"] * current_values["total_assets"]
                    cash_available = max(0, portfolio['cash'] - cash_min_level)
                    cash_to_use = min(value_to_buy_nasdaq, cash_available)
                    if cash_to_use > 0:
                        logging.info(f"- 步骤1: 使用 {cash_to_use:,.2f}元 现金买入纳指ETF")
                        shares_to_buy = math.floor((cash_to_use / (1 + config["commission_rate"])) / prices['nasdaq'] / 100) * 100
                        if _buy(portfolio, 'nasdaq', shares_to_buy, prices['nasdaq']):
                            temp_values = update_portfolio_values(portfolio, prices)
                            value_to_buy_nasdaq = max(0, target_etf_value - temp_values["nasdaq_value"])
                    if value_to_buy_nasdaq > 0 and portfolio['dividend_shares'] > 0:
                        logging.info(f"- 步骤2: 现金不足，需卖出红利ETF筹集约 {value_to_buy_nasdaq:,.2f}元")
                        value_to_sell_dividend = value_to_buy_nasdaq / (1 - config["commission_rate"])
                        dividend_shares_to_sell = min(math.ceil(value_to_sell_dividend / prices['dividend'] / 100) * 100, portfolio['dividend_shares'])
                        proceeds = _sell(portfolio, 'dividend', dividend_shares_to_sell, prices['dividend'])
                        if proceeds > 0:
                            shares_to_buy = math.floor((proceeds / (1 + config["commission_rate"])) / prices['nasdaq'] / 100) * 100
                            _buy(portfolio, 'nasdaq', shares_to_buy, prices['nasdaq'])
                current_values = update_portfolio_values(portfolio, prices)

        results.append({
            "date": date,
            "total_assets": current_values["total_assets"],
            "cash": portfolio['cash'],
            "nasdaq_etf_value": current_values["nasdaq_value"],
            "dividend_etf_value": current_values["dividend_value"],
        })
        logging.info(f"盘后状态: 总资产 {current_values['total_assets']:,.2f}, 现金 {portfolio['cash']:,.2f}, "
                     f"纳指持仓 {portfolio['nasdaq_shares']}股, 红利持仓 {portfolio['dividend_shares']}股")

    return pd.DataFrame(results)

# ==============================================================================
# 模块4: 分析与绘图 (V9 新增)
# ==============================================================================
def _save_and_plot(df: pd.DataFrame, title: str, ylabel: str, filename_prefix: str, output_dir: str, y_format=None, secondary_y_info=None):
    """通用保存CSV和绘图函数"""
    # 保存数据
    csv_path = os.path.join(output_dir, f"{filename_prefix}.csv")
    df.to_csv(csv_path)
    logging.info(f"分析数据已保存至: {csv_path}")

    # 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # 主Y轴
    if secondary_y_info:
        # 绘制主Y轴的线
        for col in df.columns:
            if col != secondary_y_info['column']:
                ax1.plot(df.index, df[col], label=col)
    else:
        df.plot(ax=ax1, legend=True)

    ax1.set_title(title, fontsize=18, fontweight='bold')
    ax1.set_xlabel('Date' if isinstance(df.index, pd.DatetimeIndex) else df.index.name, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    if y_format:
        ax1.yaxis.set_major_formatter(y_format)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # 副Y轴
    if secondary_y_info:
        ax2 = ax1.twinx()
        ax2.plot(df.index, df[secondary_y_info['column']], color=secondary_y_info['color'], linestyle='--', label=secondary_y_info['label'])
        ax2.set_ylabel(secondary_y_info['ylabel'], fontsize=12)
        if secondary_y_info.get('y_format'):
            ax2.yaxis.set_major_formatter(secondary_y_info['y_format'])
        ax2.legend(loc='upper right')

    plt.tight_layout()
    
    # 保存图表
    img_path = os.path.join(output_dir, f"{filename_prefix}.png")
    try:
        plt.savefig(img_path, dpi=300)
        logging.info(f"图表已保存至: {img_path}")
    except Exception as e:
        logging.error(f"保存图表失败: {e}")
    plt.show()

def analyze_and_plot_capital_curve(results_df: pd.DataFrame, config: dict):
    """绘制资金曲线图"""
    logging.info("正在分析和绘制资金曲线...")
    df_to_plot = results_df[['total_assets']]
    
    _save_and_plot(
        df=df_to_plot,
        title='Portfolio Value Over Time (Backtest)',
        ylabel='Total Assets (CNY)',
        filename_prefix='capital_curve',
        output_dir=config['output_dir'],
        y_format=mticker.StrMethodFormatter('{x:,.0f}')
    )

def analyze_and_plot_drawdown(results_df: pd.DataFrame, config: dict):
    """计算并绘制最大回撤"""
    logging.info("正在分析和绘制最大回撤...")
    
    analysis_df = pd.DataFrame(index=results_df.index)
    analysis_df['total_assets'] = results_df['total_assets']
    analysis_df['peak'] = analysis_df['total_assets'].expanding().max()
    analysis_df['drawdown_ratio'] = (analysis_df['total_assets'] - analysis_df['peak']) / analysis_df['peak']
    
    # 找到最大回撤点
    max_drawdown_date = analysis_df['drawdown_ratio'].idxmin()
    max_drawdown_value = analysis_df['drawdown_ratio'].min()
    logging.info(f"历史最大回撤: {max_drawdown_value:.2%} (发生在 {max_drawdown_date.strftime('%Y-%m-%d')})")

    _save_and_plot(
        df=analysis_df[['drawdown_ratio']],
        title='Portfolio Maximum Drawdown Ratio Over Time',
        ylabel='Drawdown Ratio',
        filename_prefix='drawdown_analysis',
        output_dir=config['output_dir'],
        y_format=mticker.PercentFormatter(1.0)
    )

def analyze_and_plot_holding_period(results_df: pd.DataFrame, config: dict):
    """计算并绘制持有期分析图"""
    logging.info("正在进行持有期分析 (这可能需要一些时间)...")
    
    total_assets = results_df['total_assets']
    max_days = min(config['max_holding_days'], len(total_assets) - 1)
    
    analysis_data = []
    for d in range(1, max_days + 1):
        # 计算持有d天的收益率序列
        returns_d = (total_assets / total_assets.shift(d)) - 1
        returns_d.dropna(inplace=True)
        
        if not returns_d.empty:
            analysis_data.append({
                'holding_days': d,
                'max_return': returns_d.max(),
                'min_return': returns_d.min(),
                'win_rate': (returns_d > 0).sum() / len(returns_d) if len(returns_d) > 0 else 0
            })
    
    if not analysis_data:
        logging.warning("无法进行持有期分析，数据不足。")
        return

    analysis_df = pd.DataFrame(analysis_data).set_index('holding_days')
    
    _save_and_plot(
        df=analysis_df,
        title=f'Holding Period Analysis (1 to {max_days} days)',
        ylabel='Return Rate',
        filename_prefix='holding_period_analysis',
        output_dir=config['output_dir'],
        y_format=mticker.PercentFormatter(1.0),
        secondary_y_info={
            'column': 'win_rate',
            'label': 'Win Rate',
            'ylabel': 'Win Rate',
            'color': 'green',
            'y_format': mticker.PercentFormatter(1.0)
        }
    )

def analyze_and_plot_covariance(data_df: pd.DataFrame, config: dict):
    """计算并绘制ETF的扩张窗口协方差"""
    logging.info("正在分析和绘制ETF扩张协方差...")
    
    etf_names = list(config['etf_codes'].keys())
    etf1_name, etf2_name = etf_names[0], etf_names[1]
    
    # 使用后复权收盘价计算日收益率
    etf1_hfq_col = f'close_hfq_{etf1_name}'
    etf2_hfq_col = f'close_hfq_{etf2_name}'
    
    if etf1_hfq_col not in data_df.columns or etf2_hfq_col not in data_df.columns:
        logging.error("原始数据中缺少后复权价格列，无法计算协方差。")
        return
        
    returns_etf1 = data_df[etf1_hfq_col].pct_change()
    returns_etf2 = data_df[etf2_hfq_col].pct_change()
    
    # 计算扩张窗口协方差
    expanding_cov = returns_etf1.expanding().cov(returns_etf2)
    analysis_df = pd.DataFrame(expanding_cov).rename(columns={0: 'expanding_covariance'})
    analysis_df.dropna(inplace=True)

    _save_and_plot(
        df=analysis_df,
        title=f'Expanding Covariance between {etf1_name.upper()} and {etf2_name.upper()} (HFQ Returns)',
        ylabel='Covariance',
        filename_prefix='covariance_analysis',
        output_dir=config['output_dir']
    )

# ==============================================================================
# 模块5: 执行与输出
# ==============================================================================
if __name__ == "__main__":
    setup_logging()
    
    # 创建输出目录
    output_dir = CONFIG['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"已创建输出目录: {output_dir}")

    master_data = get_prepared_data(CONFIG)
    if not master_data.empty:
        results_df = run_backtest(CONFIG, master_data)
        if not results_df.empty:
            results_df.set_index('date', inplace=True)
            
            # 保存主要回测结果
            main_results_path = os.path.join(output_dir, "backtest_daily_results.csv")
            results_df.to_csv(main_results_path)
            logging.info(f"回测完成！每日结果已保存至: {main_results_path}")
            
            # 计算并打印摘要
            initial_assets = CONFIG["initial_capital"]
            final_assets = results_df['total_assets'].iloc[-1]
            total_return = (final_assets / initial_assets) - 1
            duration_years = (results_df.index[-1] - results_df.index[0]).days / 365.25
            annualized_return = (1 + total_return) ** (1 / duration_years) - 1 if duration_years > 0 else 0
            
            # 计算最大回撤用于摘要
            peak = results_df['total_assets'].expanding().max()
            drawdown = (results_df['total_assets'] - peak) / peak
            max_drawdown = drawdown.min()

            summary = "\n--- 回测结果摘要 ---\n"
            summary += f"起始日期: {results_df.index[0].strftime('%Y-%m-%d')}\n"
            summary += f"结束日期: {results_df.index[-1].strftime('%Y-%m-%d')}\n"
            summary += f"回测时长: {duration_years:.2f} 年\n"
            summary += f"初始资产: {initial_assets:,.2f} 元\n"
            summary += f"最终资产: {final_assets:,.2f} 元\n"
            summary += f"总收益率: {total_return:.2%}\n"
            summary += f"年化收益率: {annualized_return:.2%}\n"
            summary += f"最大回撤: {max_drawdown:.2%}\n"
            summary += "--------------------"
            logging.info(summary)

            # --- 执行所有分析和绘图 ---
            analyze_and_plot_capital_curve(results_df, CONFIG)
            analyze_and_plot_drawdown(results_df, CONFIG)
            analyze_and_plot_holding_period(results_df, CONFIG)
            analyze_and_plot_covariance(master_data, CONFIG)
            
            logging.info("\n所有分析和绘图已完成！")

        else:
            logging.error("回测未产生任何结果。")
