# trade_manager/service/m_distribution_reporter.py (替换整个文件)

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
from django.db.models import Q

from common.models.backtest_logs import MDistributionBacktestLog
from data_manager.service.email_handler import EmailHandler

logger = logging.getLogger(__name__)

class MDistributionReporter:
    """
    M值胜率分布回测的报告生成与发送器 (V2 - 多策略对比版)。
    """
    def __init__(self, backtest_run_id: str, date_range_text: str):
        self.backtest_run_id = backtest_run_id
        self.date_range_text = date_range_text
        self.email_handler = EmailHandler()
        self.recipients = ['876858298@qq.com']

    def _fetch_data_for_strategy(self, query_filter: Q) -> pd.DataFrame:
        """从数据库获取指定策略的回测结果"""
        base_query = MDistributionBacktestLog.objects.filter(
            backtest_run_id=self.backtest_run_id
        ).exclude(
            exit_reason=MDistributionBacktestLog.ExitReason.END_OF_PERIOD
        )
        
        logs = base_query.filter(query_filter)
        
        if not logs.exists():
            return pd.DataFrame()
        
        return pd.DataFrame.from_records(logs.values())

    def _analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """对单个策略的数据进行分箱和统计分析"""
        if df.empty:
            return pd.DataFrame()

        bins = np.arange(-1.0, 1.1, 0.1)
        labels = [f"{i:.1f} to {i+0.1:.1f}" for i in bins[:-1]]
        
        df['m_interval'] = pd.cut(df['m_value_at_plan'].astype(float), bins=bins, labels=labels, right=False)

        def agg_func(group):
            total_trades = len(group)
            if total_trades == 0:
                return pd.Series({
                    'total_trades': 0, 'win_rate': 0, 'expected_return': 0
                })
            
            win_trades = (group['exit_reason'] == 'TAKE_PROFIT').sum()
            win_rate = win_trades / total_trades
            
            avg_tp_rate = np.nan_to_num(group[group['exit_reason'] == 'TAKE_PROFIT']['preset_take_profit_rate'].astype(float).mean())
            avg_sl_rate = np.nan_to_num(group[group['exit_reason'] == 'STOP_LOSS']['preset_stop_loss_rate'].astype(float).mean())
            
            expected_return = (win_rate * avg_tp_rate - (1 - win_rate) * avg_sl_rate)
            
            return pd.Series({
                'total_trades': total_trades,
                'win_rate': win_rate,
                'expected_return': expected_return
            })

        analysis_df = df.groupby('m_interval', observed=True).apply(agg_func, include_groups=False).reset_index()
        return analysis_df

    def _generate_combined_plots_base64(self, all_analysis_results: dict) -> tuple[str, str]:
        """生成包含多条曲线的组合图表"""
        if not all_analysis_results:
            return "", ""

        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass

        # 胜率组合图
        fig_win, ax_win = plt.subplots(figsize=(14, 7))
        for name, df in all_analysis_results.items():
            if not df.empty:
                ax_win.plot(df['m_interval'], df['win_rate'], marker='o', linestyle='-', label=name)
        
        ax_win.set_title('各策略胜率 vs M值', fontsize=16)
        ax_win.set_xlabel('M值区间', fontsize=12)
        ax_win.set_ylabel('胜率', fontsize=12)
        ax_win.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
        ax_win.grid(True, linestyle='--', alpha=0.6)
        ax_win.legend()
        plt.setp(ax_win.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        buf_win = io.BytesIO()
        fig_win.savefig(buf_win, format='png', dpi=120)
        win_rate_b64 = base64.b64encode(buf_win.getvalue()).decode('utf-8')
        plt.close(fig_win)

        # 期望收益组合图
        fig_exp, ax_exp = plt.subplots(figsize=(14, 7))
        for name, df in all_analysis_results.items():
            if not df.empty:
                ax_exp.plot(df['m_interval'], df['expected_return'], marker='s', linestyle='--', label=name)

        ax_exp.set_title('各策略期望收益率 vs M值', fontsize=16)
        ax_exp.set_xlabel('M值区间', fontsize=12)
        ax_exp.set_ylabel('期望收益率', fontsize=12)
        ax_exp.yaxis.set_major_formatter(plt.FuncFormatter('{:.2%}'.format))
        ax_exp.axhline(0, color='grey', linestyle=':', linewidth=1)
        ax_exp.grid(True, linestyle='--', alpha=0.6)
        ax_exp.legend()
        plt.setp(ax_exp.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        buf_exp = io.BytesIO()
        fig_exp.savefig(buf_exp, format='png', dpi=120)
        exp_return_b64 = base64.b64encode(buf_exp.getvalue()).decode('utf-8')
        plt.close(fig_exp)

        return win_rate_b64, exp_return_b64

    def _format_html_content(self, all_analysis_results: dict, plot1_b64: str, plot2_b64: str) -> str:
        """将所有内容整合成HTML邮件"""
        
        # --- 生成所有表格 ---
        tables_html = ""
        for name, df in all_analysis_results.items():
            tables_html += f"<h2>{name} - 详细数据统计</h2>"
            if df.empty:
                tables_html += "<p>该策略无有效交易数据。</p>"
                continue

            df_display = df.copy()
            df_display['win_rate'] = df_display['win_rate'].apply(lambda x: f"{x:.2%}")
            df_display['expected_return'] = df_display['expected_return'].apply(lambda x: f"{x:.2%}")
            df_display.rename(columns={
                'm_interval': 'M值区间',
                'total_trades': '总交易数',
                'win_rate': '胜率',
                'expected_return': '期望收益率',
            }, inplace=True)
            
            tables_html += df_display.to_html(index=False, classes='styled-table', border=0)

        # --- 最终HTML模板 ---
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>M值胜率分布回测报告</title>
            <style>
                body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 20px; background-color: #f8f9fa; }}
                h1, h2 {{ color: #0056b3; border-bottom: 2px solid #e9ecef; padding-bottom: 8px; }}
                .styled-table {{ border-collapse: collapse; margin: 25px 0; font-size: 0.9em; min-width: 600px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); }}
                .styled-table thead tr {{ background-color: #007bff; color: #ffffff; text-align: left; }}
                .styled-table th, .styled-table td {{ padding: 12px 15px; }}
                .styled-table tbody tr {{ border-bottom: 1px solid #dddddd; }}
                .styled-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
                .plot-container {{ text-align: center; margin-top: 20px; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>M值胜率分布回测报告</h1>
            <h2>日期区间: {self.date_range_text}</h2>
            
            <h2>组合图表分析</h2>
            <div class="plot-container">
                <h3>各策略胜率 vs M值</h3>
                <img src="data:image/png;base64,{plot1_b64}" alt="Win Rate Plot">
            </div>
            <div class.plot-container">
                <h3>各策略期望收益率 vs M值</h3>
                <img src="data:image/png;base64,{plot2_b64}" alt="Expected Return Plot">
            </div>
            
            {tables_html}
        </body>
        </html>
        """
        return html

    def generate_and_send_report(self):
        """生成并发送报告邮件的主方法"""
        logger.info(f"[{self.backtest_run_id}] 开始生成M值分布回测报告 (多策略版)...")
        try:
            # 1. 获取策略分组
            strategy_groups_raw = MDistributionBacktestLog.objects.filter(
                backtest_run_id=self.backtest_run_id
            ).values_list('one_stratage_mode', flat=True).distinct()

            if not strategy_groups_raw:
                logger.warning(f"[{self.backtest_run_id}] 数据库中无任何日志，无法生成报告。")
                return

            # 2. 整理和排序分组
            groups_to_process = sorted(
                [g if g is not None else 'M_DYNAMIC' for g in strategy_groups_raw],
                key=lambda x: (x != 'M_DYNAMIC', x)
            )

            # 3. 循环获取数据并分析
            all_analysis_results = {}
            for strategy_key in groups_to_process:
                display_name = "M动态策略" if strategy_key == 'M_DYNAMIC' else f"单策略 - {strategy_key}"
                query_filter = Q(one_stratage_mode__isnull=True) if strategy_key == 'M_DYNAMIC' else Q(one_stratage_mode=strategy_key)
                
                strategy_df = self._fetch_data_for_strategy(query_filter)
                analysis_df = self._analyze_data(strategy_df)
                all_analysis_results[display_name] = analysis_df

            # 4. 生成图表和HTML
            plot1_b64, plot2_b64 = self._generate_combined_plots_base64(all_analysis_results)
            html_content = self._format_html_content(all_analysis_results, plot1_b64, plot2_b64)
            
            subject = f"M值胜率分布回测报告 (多策略版) - {datetime.now().strftime('%Y-%m-%d')}"
            
            self.email_handler.send_email(
                recipients=self.recipients,
                subject=subject,
                html_content=html_content
            )
            logger.info(f"[{self.backtest_run_id}] 多策略回测报告邮件已成功发送。")
        except Exception as e:
            logger.error(f"[{self.backtest_run_id}] 生成或发送多策略报告时失败: {e}", exc_info=True)
