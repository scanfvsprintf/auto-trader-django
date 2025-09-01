# ==============================================================================
# 文件 3/5: trade_manager/service/backtest_reporter.py (新增)
# 描述: 负责生成和发送回测邮件报告的模块。
# ==============================================================================
import base64
import io
import logging
from datetime import date
from decimal import Decimal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
from django.db import connections

from common.models.backtest_logs import BacktestDailyLog, BacktestOperationLog
from common.models.positions import Position
from data_manager.service.email_handler import EmailHandler

logger = logging.getLogger(__name__)

class BacktestReporter:
    """
    回测报告生成与发送器。
    """
    def __init__(self, schema_name: str, start_date: date, current_date: date, initial_capital: Decimal):
        self.schema_name = schema_name
        self.start_date = start_date
        self.current_date = current_date
        self.initial_capital = initial_capital
        self.email_handler = EmailHandler()
        self.recipients = ['876858298@qq.com','850696281@qq.com']#,'285173686@qq.com','850696281@qq.com'
    def _execute_query(self, query: str, params: list = None) -> list[dict]:
        """在指定 schema 中执行原生 SQL 查询并返回结果"""
        with connections['default'].cursor() as cursor:
            cursor.execute(f'SET search_path TO "{self.schema_name}", public;')
            cursor.execute(query, params or [])
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _generate_report_data(self) -> dict:
        """准备邮件报告所需的所有数据"""
        data = {}

        # 1. 关键指标
        daily_logs = self._execute_query(
            f"SELECT trade_date, total_assets FROM {BacktestDailyLog._meta.db_table} ORDER BY trade_date"
        )
        df_daily = pd.DataFrame(daily_logs)
        df_daily['total_assets'] = df_daily['total_assets'].astype(float)
        
        # 胜率
        sell_ops = self._execute_query(
            f"SELECT exit_reason FROM {BacktestOperationLog._meta.db_table} WHERE direction = 'SELL'"
        )
        if sell_ops:
            total_sells = len(sell_ops)
            profit_sells = sum(1 for op in sell_ops if op['exit_reason'] == 'TAKE_PROFIT')
            data['win_rate'] = profit_sells / total_sells if total_sells > 0 else 0.0
        else:
            data['win_rate'] = 0.0
        
        # 最大回撤
        df_daily['peak'] = df_daily['total_assets'].cummax()
        df_daily['drawdown'] = (df_daily['total_assets'] - df_daily['peak']) / df_daily['peak']
        data['max_drawdown'] = df_daily['drawdown'].min() if not df_daily.empty else 0.0

        # 年化收益率
        final_assets = float(df_daily['total_assets'].iloc[-1]) if not df_daily.empty else float(self.initial_capital)
        days_passed = (self.current_date - self.start_date).days
        if days_passed > 0:
            data['annualized_return'] = ((final_assets / float(self.initial_capital)) ** (365.0 / days_passed)) - 1
        else:
            data['annualized_return'] = 0.0

        # 2. 资金曲线图数据
        data['plot_data'] = self._execute_query(
            f"SELECT trade_date, total_assets, market_m_value FROM {BacktestDailyLog._meta.db_table} ORDER BY trade_date"
        )

        # 3. 当前持仓
        data['current_holdings'] = self._execute_query(
            f"""
            SELECT p.stock_code, si.stock_name, p.entry_price, p.quantity, 
                   p.current_take_profit, p.current_stop_loss, dq.close as current_price
            FROM {Position._meta.db_table} p
            JOIN public.tb_stock_info si ON p.stock_code = si.stock_code
            LEFT JOIN public.tb_daily_quotes dq ON p.stock_code = dq.stock_code AND dq.trade_date = %s
            WHERE p.status = 'open'
            """, [self.current_date]
        )
        for h in data['current_holdings']:
            entry_price = h['entry_price']
            if entry_price and entry_price > 0:
                # 新的计算方式：将止盈/止损价表示为成本价的百分比
                h['profit_level_pct'] = h['current_take_profit'] / entry_price
                h['loss_level_pct'] = h['current_stop_loss'] / entry_price
            else:
                # 处理 entry_price 无效的情况
                h['profit_level_pct'] = Decimal('0.0')
                h['loss_level_pct'] = Decimal('0.0')

        # 4. 收益排名
        all_ops = self._execute_query(f"SELECT stock_code, stock_name, direction, amount FROM {BacktestOperationLog._meta.db_table}")
        profits = {}
        for op in all_ops:
            key = (op['stock_code'], op['stock_name'])
            if op['direction'] == 'BUY':
                profits[key] = profits.get(key, 0) - op['amount']
            else: # SELL
                profits[key] = profits.get(key, 0) + op['amount']
        # 总收益 = 已实现盈亏 + 未实现盈亏
        #        = (卖出总额 - 买入总额) + (当前市值 - 持仓成本)
        #        = (卖出总额) - (已平仓部分的买入成本) + (当前市值)
        # 之前的循环已经计算了 (卖出总额 - 全部买入成本)，所以我们只需加上当前市值即可。
        for holding in data['current_holdings']:
            key = (holding['stock_code'], holding['stock_name'])
            
            # 处理当天可能没有行情数据的情况，若无当前价则按入场价计算，浮动盈亏为0
            current_price = holding['current_price'] or holding['entry_price']
            
            # 计算当前持仓的总市值
            current_market_value = holding['quantity'] * current_price
            
            # 将当前市值加到该股票的累计收益中
            profits[key] = profits.get(key, 0) + current_market_value
        profit_list = [{'stock_code': k[0], 'stock_name': k[1], 'profit': v} for k, v in profits.items()]
        data['profit_ranking'] = sorted(profit_list, key=lambda x: x['profit'], reverse=True)

        return data

    def _generate_plot_base64(self, plot_data: list[dict]) -> str:
        if not plot_data:
            return ""
        
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            pass
        
        try:
            df = pd.DataFrame(plot_data)
            # 确保数据类型正确
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['total_assets'] = pd.to_numeric(df['total_assets'])
            df['market_m_value'] = pd.to_numeric(df['market_m_value'])

            if df.empty:
                return ""

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax1 = plt.subplots(figsize=(14, 7))

            # 绘制主曲线
            ax1.plot(df['trade_date'], df['total_assets'], color='dodgerblue', label='money', linewidth=2)
            ax1.set_xlabel('date', fontsize=12)
            ax1.set_ylabel('money', color='dodgerblue', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='dodgerblue')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

            # 绘制副坐标轴曲线
            ax2 = ax1.twinx()
            ax2.plot(df['trade_date'], df['market_m_value'], color='coral', linestyle='--', label='M', alpha=0.7)
            ax2.set_ylabel('M', color='coral', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='coral')
            ax2.axhline(0, color='grey', linestyle=':', linewidth=1)

            # ======================= 主要修改点 =======================
            # 1. 更健壮和简化的X轴刻度逻辑
            num_days = (df['trade_date'].max() - df['trade_date'].min()).days
            
            if num_days <= 60:  # 2个月以内，按周显示
                locator = mdates.WeekdayLocator(byweekday=mdates.MO)
                formatter = mdates.DateFormatter('%m-%d')
            elif num_days <= 365 * 2: # 2年以内，按季度显示
                locator = mdates.MonthLocator(interval=3)
                formatter = mdates.DateFormatter('%Y-%m')
            elif num_days <= 365 * 5: # 5年以内，按半年显示
                locator = mdates.MonthLocator(interval=6)
                formatter = mdates.DateFormatter('%Y-%m')
            else:  # 超过5年，按年显示
                locator = mdates.YearLocator()
                formatter = mdates.DateFormatter('%Y')
            
            ax1.xaxis.set_major_locator(locator)
            ax1.xaxis.set_major_formatter(formatter)

            # 2. 移除 fig.autofmt_xdate()，并手动设置标签旋转，避免冲突
            plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
            # ========================================================

            fig.suptitle('Money-M(t)', fontsize=16, weight='bold')
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            
            # 使用 tight_layout 替代
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # 保存图像到内存
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        finally:
            pass



    def _format_html_content(self, data: dict, plot_base64: str) -> str:
        
        # ======================= 修复颜色逻辑 =======================
        # 修复：盈利 (value > 0) 为绿色，亏损 (value < 0) 为红色
        def get_row_style(value):
            if value > 0:
                return 'style="background-color: #e9f5e9; color: #1e7e34;"'  # 绿色背景，深绿色文字
            elif value < 0:
                return 'style="background-color: #fdeaea; color: #c82333;"'  # 红色背景，深红色文字
            else:
                return '' # 中性
        # ==========================================================
        # Part 1: Key Metrics
        html = f"""
        <h2>关键指标 (截至 {self.current_date.strftime('%Y-%m-%d')})</h2>
        <table class="summary-table">
            <tr>
                <th>胜率</th><td>{data['win_rate']:.2%}</td>
                <th>最大回撤</th><td style="color: #c82333;">{data['max_drawdown']:.2%}</td>
                <th>年化收益率</th><td>{data['annualized_return']:.2%}</td>
            </tr>
        </table>
        """
        # Part 2: Plot
        html += f"""
        <h2>资金与M值变化趋势</h2>
        <div style="text-align: center;">
            <img src="data:image/png;base64,{plot_base64}" alt="Performance Chart" style="max-width: 100%;">
        </div>
        """
        # Part 3: Current Holdings
        html += "<h2>当前持仓情况</h2>"
        if data['current_holdings']:
            html += """
            <table class="data-table">
                <thead><tr><th>股票代码</th><th>股票名称</th><th>入场价</th><th>当前价</th><th>浮动盈亏</th><th>止盈价格</th><th>止损价格</th><th>止盈线位置</th><th>止损线位置</th></tr></thead>
                <tbody>
            """
            for h in data['current_holdings']:
                current_price = h['current_price'] or h['entry_price']
                profit_loss = current_price - h['entry_price']
                profit_loss_rate = (current_price / h['entry_price'] - 1) if h['entry_price'] else 0
                style = get_row_style(profit_loss)
                profit_level_str = f"{h['profit_level_pct']:.2%}"
                loss_level_str = f"{h['loss_level_pct']:.2%}"
                html += f"""
                <tr {style}>
                    <td>{h['stock_code']}</td>
                    <td>{h['stock_name']}</td>
                    <td>{h['entry_price']:.2f}</td>
                    <td>{current_price:.2f}</td>
                    <td>{profit_loss_rate:.2%}</td>
                    <td>{h['current_take_profit']:.2f}</td>
                    <td>{h['current_stop_loss']:.2f}</td>
                    <td style="color: #c82333;">{profit_level_str}</td>
                    <td style="color: #1e7e34;">{loss_level_str}</td>
                </tr>
                """
            html += "</tbody></table>"
        else:
            html += "<p>当前无持仓。</p>"
        # Part 4: Profit Ranking
        html += "<h2>各股累计收益排名</h2>"
        if data['profit_ranking']:
            html += """
            <table class="data-table">
                <thead><tr><th>排名</th><th>股票代码</th><th>股票名称</th><th>累计收益(元)</th></tr></thead>
                <tbody>
            """
            for i, p in enumerate(data['profit_ranking'], 1):
                # 这里复用上面修改好的颜色逻辑
                style = get_row_style(p['profit'])
                html += f"""
                <tr {style}>
                    <td>{i}</td>
                    <td>{p['stock_code']}</td>
                    <td>{p['stock_name']}</td>
                    <td>{p['profit']:,.2f}</td>
                </tr>
                """
            html += "</tbody></table>"
        else:
            html += "<p>暂无已平仓的交易。</p>"
        # Final HTML structure (保持不变)
        final_html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>回测报告</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }}
                h2 {{ color: #0056b3; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; margin-top: 30px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .summary-table th {{ width: 15%; background-color: #e9ecef; }}
                .summary-table td {{ font-weight: bold; font-size: 1.1em; }}
                .data-table tbody tr:hover {{ background-color: #f1f1f1; }}
            </style>
        </head>
        <body>
            <h1>回测进度报告: {self.start_date}~{self.current_date}回测</h1>
            {html}
        </body>
        </html>
        """
        return final_html

    def send_report(self):
        """生成并发送报告邮件"""
        logger.info(f"[{self.schema_name}] 正在生成截至 {self.current_date} 的回测报告...")
        try:
            report_data = self._generate_report_data()
            plot_base64 = self._generate_plot_base64(report_data.get('plot_data', []))
            html_content = self._format_html_content(report_data, plot_base64)
            subject = f"回测报告 ({self.start_date}~{self.current_date}) - {self.current_date.strftime('%Y-%m-%d')}"
            
            self.email_handler.send_email(
                recipients=self.recipients,
                subject=subject,
                html_content=html_content
            )
            logger.info(f"[{self.start_date}~{self.current_date}] 回测报告邮件已成功发送。")
        except Exception as e:
            logger.error(f"[{self.start_date}~{self.current_date}] 生成或发送回测报告时失败: {e}", exc_info=True)

