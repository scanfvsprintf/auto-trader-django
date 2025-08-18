# trade_manager/service/m_distribution_reporter.py (新文件)

import logging
import base64
import io
from decimal import Decimal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import date, timedelta,datetime
from django.db import connections

from common.models.backtest_logs import MDistributionBacktestLog
from data_manager.service.email_handler import EmailHandler
from .db_utils import use_backtest_schema

logger = logging.getLogger(__name__)

class MDistributionReporter:
    """
    M值胜率分布回测的报告生成与发送器。
    """
    def __init__(self, backtest_run_id: str,date_range_text:str):
        self.backtest_run_id = backtest_run_id
        self.date_range_text=date_range_text
        self.email_handler = EmailHandler()
        self.recipients = ['876858298@qq.com']

    def _fetch_data(self) -> pd.DataFrame:
        """从数据库获取回测结果"""
        logs = MDistributionBacktestLog.objects.filter(
            backtest_run_id=self.backtest_run_id
        ).exclude(
            exit_reason=MDistributionBacktestLog.ExitReason.END_OF_PERIOD
        )
        if not logs.exists():
            return pd.DataFrame()
        
        return pd.DataFrame.from_records(logs.values())

    def _analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """对数据进行分箱和统计分析"""
        if df.empty:
            return pd.DataFrame()

        # 定义M值的分箱区间
        bins = np.arange(-1.0, 1.1, 0.1)
        labels = [f"{i:.1f} to {i+0.1:.1f}" for i in bins[:-1]]
        
        df['m_interval'] = pd.cut(df['m_value_at_plan'].astype(float), bins=bins, labels=labels, right=False)

        # 统计分析
        def agg_func(group):
            total_trades = len(group)
            win_trades = (group['exit_reason'] == 'TAKE_PROFIT').sum()
            loss_trades = (group['exit_reason'] == 'STOP_LOSS').sum()
            
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
            # 计算期望收益率
            avg_tp_rate = np.nan_to_num(group[group['exit_reason'] == 'TAKE_PROFIT']['preset_take_profit_rate'].mean())
            avg_sl_rate = np.nan_to_num(group[group['exit_reason'] == 'STOP_LOSS']['preset_stop_loss_rate'].mean())
            
            expected_return = (win_rate * avg_tp_rate - (1 - win_rate) * avg_sl_rate) if total_trades > 0 else 0

            # 计算因子比例
            dna_list = group['strategy_dna'].str.split('|', expand=True)
            dna_proportions = {}
            for col in dna_list.columns:
                parts = dna_list[col].str.split(':', expand=True)
                if not parts.empty:
                    strategy_name = parts.iloc[0, 0]
                    avg_proportion = pd.to_numeric(parts[1], errors='coerce').mean()
                    dna_proportions[f'prop_{strategy_name}'] = avg_proportion

            return pd.Series({
                'total_trades': total_trades,
                'win_trades': win_trades,
                'loss_trades': loss_trades,
                'win_rate': win_rate,
                'expected_return': expected_return,
                **dna_proportions
            })

        analysis_df = df.groupby('m_interval',observed=True).apply(agg_func, include_groups=False).reset_index()
        return analysis_df

    def _generate_plots_base64(self, analysis_df: pd.DataFrame) -> tuple[str, str]:
        """生成胜率和期望收益图表的Base64编码"""
        if analysis_df.empty:
            return "", ""

        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass

        # 胜率图
        fig_win, ax_win = plt.subplots(figsize=(12, 6))
        ax_win.plot(analysis_df['m_interval'], analysis_df['win_rate'], marker='o', linestyle='-', color='b')
        ax_win.set_title('Winning Rate/M Value', fontsize=16)
        ax_win.set_xlabel('M Value', fontsize=12)
        ax_win.set_ylabel('Winning Rate', fontsize=12)
        ax_win.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
        ax_win.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf_win = io.BytesIO()
        fig_win.savefig(buf_win, format='png', dpi=100)
        win_rate_b64 = base64.b64encode(buf_win.getvalue()).decode('utf-8')
        plt.close(fig_win)

        # 期望收益图
        fig_exp, ax_exp = plt.subplots(figsize=(12, 6))
        ax_exp.plot(analysis_df['m_interval'], analysis_df['expected_return'], marker='s', linestyle='--', color='g')
        ax_exp.set_title('Expected Rate/M Value', fontsize=16)
        ax_exp.set_xlabel('M Value', fontsize=12)
        ax_exp.set_ylabel('‌Expected Rate of Return‌', fontsize=12)
        ax_exp.yaxis.set_major_formatter(plt.FuncFormatter('{:.2%}'.format))
        ax_exp.axhline(0, color='grey', linestyle=':', linewidth=1)
        ax_exp.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf_exp = io.BytesIO()
        fig_exp.savefig(buf_exp, format='png', dpi=100)
        exp_return_b64 = base64.b64encode(buf_exp.getvalue()).decode('utf-8')
        plt.close(fig_exp)

        return win_rate_b64, exp_return_b64

    def _format_html_content(self, analysis_df: pd.DataFrame, plot1_b64: str, plot2_b64: str) -> str:
        """将所有内容整合成HTML邮件"""
        # 格式化DataFrame以便在HTML中显示
        df_display = analysis_df.copy()
        df_display['win_rate'] = df_display['win_rate'].apply(lambda x: f"{x:.2%}")
        df_display['expected_return'] = df_display['expected_return'].apply(lambda x: f"{x:.2%}")
        for col in [c for c in df_display.columns if c.startswith('prop_')]:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")
        
        # 重命名列以提高可读性
        df_display.rename(columns={
            'm_interval': 'M值区间',
            'total_trades': '总交易数',
            'win_trades': '止盈数',
            'loss_trades': '止损数',
            'win_rate': '胜率',
            'expected_return': '期望收益率',
            'prop_MT': '趋势动能占比',
            'prop_BO': '强势突破占比',
            'prop_MR': '均值回归占比',
            'prop_QD': '质量防御占比'
        }, inplace=True)
        
        html_table = df_display.to_html(index=False, classes='styled-table', border=0)

        # HTML模板
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>M值胜率分布回测报告</title>
            <style>
                body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 20px; }}
                h1, h2 {{ color: #0056b3; }}
                .styled-table {{ border-collapse: collapse; margin: 25px 0; font-size: 0.9em; min-width: 400px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); }}
                .styled-table thead tr {{ background-color: #009879; color: #ffffff; text-align: left; }}
                .styled-table th, .styled-table td {{ padding: 12px 15px; }}
                .styled-table tbody tr {{ border-bottom: 1px solid #dddddd; }}
                .styled-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
                .styled-table tbody tr:last-of-type {{ border-bottom: 2px solid #009879; }}
                .plot-container {{ text-align: center; margin-top: 20px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>M值胜率分布回测报告</h1>
            <h2>日期区间: {self.date_range_text}</h2>
            
            <h2>详细数据统计</h2>
            {html_table}
            
            <h2>胜率 vs M值</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{plot1_b64}" alt="Win Rate vs M-Value Plot">
            </div>
            
            <h2>期望收益 vs M值</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{plot2_b64}" alt="Expected Return vs M-Value Plot">
            </div>
        </body>
        </html>
        """
        return html

    def generate_and_send_report(self):
        """生成并发送报告邮件的主方法"""
        logger.info(f"[{self.backtest_run_id}] 开始生成M值分布回测报告...")
        try:
            data_df = self._fetch_data()
            if data_df.empty:
                logger.warning(f"[{self.backtest_run_id}] 没有找到任何有效的回测日志，无法生成报告。")
                return

            analysis_df = self._analyze_data(data_df)
            plot1_b64, plot2_b64 = self._generate_plots_base64(analysis_df)
            html_content = self._format_html_content(analysis_df, plot1_b64, plot2_b64)
            
            subject = f"M值胜率分布回测报告 - {datetime.now().strftime('%Y-%m-%d')}"
            
            self.email_handler.send_email(
                recipients=self.recipients,
                subject=subject,
                html_content=html_content
            )
            logger.info(f"[{self.backtest_run_id}] 回测报告邮件已成功发送。")
        except Exception as e:
            logger.error(f"[{self.backtest_run_id}] 生成或发送报告时失败: {e}", exc_info=True)

