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
import seaborn as sns  
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
    # ===================== 新增二维分析方法 =====================
    def _analyze_2d_distribution(self, df: pd.DataFrame) -> dict:
        """
        对 M动态策略 的数据进行二维 M值 x 选股得分 的分析。
        返回一个字典，包含用于表格和热力图的多个DataFrame。
        """
        if df.empty or 'final_score' not in df.columns:
            return {}
        df = df.copy()
        df.dropna(subset=['final_score', 'm_value_at_plan'], inplace=True)
        df['final_score'] = df['final_score'].astype(float)
        df['m_value_at_plan'] = df['m_value_at_plan'].astype(float)
        # 定义计算函数
        def calculate_metrics(group):
            win_trades = (group['exit_reason'] == 'TAKE_PROFIT').sum()
            total_trades = len(group)
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
            avg_tp = np.nan_to_num(group.loc[group['exit_reason'] == 'TAKE_PROFIT', 'preset_take_profit_rate'].astype(float).mean())
            avg_sl = np.nan_to_num(group.loc[group['exit_reason'] == 'STOP_LOSS', 'preset_stop_loss_rate'].astype(float).mean())
            expected_return = win_rate * avg_tp - (1 - win_rate) * avg_sl
            
            return pd.Series({'win_rate': win_rate, 'expected_return': expected_return})
        # --- 为HTML表格进行分析 (0.1步长) ---
        m_bins_table = np.arange(-1.0, 1.1, 0.1)
        score_bins_table = np.arange(-1.0, 1.1, 0.1)
        df['m_bin'] = pd.cut(df['m_value_at_plan'], bins=m_bins_table, right=False)
        df['score_bin'] = pd.cut(df['final_score'], bins=score_bins_table, right=False)
        
        table_grouped = df.groupby(['m_bin', 'score_bin'], observed=False).apply(calculate_metrics, include_groups=False).reset_index()
        
        # --- 为热力图进行分析 (评分使用0.01步长) ---
        score_bins_heatmap = np.arange(-1.0, 1.01, 0.01) # 步长为0.01
        df['score_bin_fine'] = pd.cut(df['final_score'], bins=score_bins_heatmap, right=False)
        
        heatmap_grouped = df.groupby(['m_bin', 'score_bin_fine'], observed=False).apply(calculate_metrics, include_groups=False).reset_index()
        
        # 创建Pivot Tables
        res = {
            'table_win_rate': table_grouped.pivot_table(index='m_bin', columns='score_bin', values='win_rate'),
            'table_return': table_grouped.pivot_table(index='m_bin', columns='score_bin', values='expected_return'),
            'heatmap_win_rate': heatmap_grouped.pivot_table(index='m_bin', columns='score_bin_fine', values='win_rate'),
            'heatmap_return': heatmap_grouped.pivot_table(index='m_bin', columns='score_bin_fine', values='expected_return')
        }
        
        # 填充NaN值为-999，以便在热力图和表格中清晰区分无数据区域
        for key in res:
            res[key].fillna(-999, inplace=True)
            
        return res
    # ===================== 新增热力图生成方法 =====================
    def _generate_heatmaps_base64(self, win_rate_df: pd.DataFrame, return_df: pd.DataFrame) -> tuple[str, str]:
        """为胜率和收益率生成两个热力图"""
        if win_rate_df.empty or return_df.empty:
            return "", ""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
        # --- 胜率热力图 ---
        fig1, ax1 = plt.subplots(figsize=(20, 8))
        sns.heatmap(
            win_rate_df.replace(-999, np.nan), ax=ax1, cmap="viridis", annot=False, # 数据量大，关闭 annot
            fmt=".1%", cbar_kws={'label': '胜率'}
        )
        ax1.set_title('胜率分布 (M值 vs 选股得分)', fontsize=16)
        ax1.set_xlabel('选股得分区间 (步长0.01)')
        ax1.set_ylabel('M值区间 (步长0.1)')
        # 简化X轴标签，每10个显示一个
        tick_labels = [f"{x.left:.2f}" for x in win_rate_df.columns]
        ax1.set_xticks(np.arange(len(tick_labels))[::10] + 0.5)
        ax1.set_xticklabels(tick_labels[::10], rotation=45, ha='right')
        plt.tight_layout()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=100)
        win_rate_b64 = base64.b64encode(buf1.getvalue()).decode('utf-8')
        plt.close(fig1)
        # --- 期望收益热力图 ---
        fig2, ax2 = plt.subplots(figsize=(20, 8))
        sns.heatmap(
            return_df.replace(-999, np.nan), ax=ax2, cmap="icefire", center=0, annot=False,
            fmt=".2%", cbar_kws={'label': '期望收益率'}
        )
        ax2.set_title('期望收益率分布 (M值 vs 选股得分)', fontsize=16)
        ax2.set_xlabel('选股得分区间 (步长0.01)')
        ax2.set_ylabel('M值区间 (步长0.1)')
        ax2.set_xticks(np.arange(len(tick_labels))[::10] + 0.5)
        ax2.set_xticklabels(tick_labels[::10], rotation=45, ha='right')
        plt.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=100)
        return_b64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
        plt.close(fig2)
        return win_rate_b64, return_b64
    def _format_html_content(self, all_analysis_results: dict, plot1_b64: str, plot2_b64: str,
                             two_dim_tables: dict, heatmap1_b64: str, heatmap2_b64: str) -> str:
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
        # ===================== 新增二维表格HTML生成 =====================
        tables_2d_html = "<h2>新增：M动态策略 - 二维统计表</h2>"
        if two_dim_tables:
            # 胜率二维表
            tables_2d_html += "<h3>胜率 (M值 vs 选股得分)</h3>"
            df_wr = two_dim_tables['table_win_rate'].replace(-999, 'N/A').applymap(lambda x: f"{x:.1%}" if isinstance(x, (float, int)) else x)
            tables_2d_html += df_wr.to_html(classes='styled-table styled-table-2d', border=0, na_rep='-')
            # 收益率二维表
            tables_2d_html += "<h3>期望收益率 (M值 vs 选股得分)</h3>"
            df_er = two_dim_tables['table_return'].replace(-999, 'N/A').applymap(lambda x: f"{x:.2%}" if isinstance(x, (float, int)) else x)
            tables_2d_html += df_er.to_html(classes='styled-table styled-table-2d', border=0, na_rep='-')
        else:
            tables_2d_html += "<p>M动态策略无数据，无法生成二维表。</p>"
        # =============================================================
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
            <!-- ===================== 新增二维热力图 ===================== -->
            <h2>热力图分析 (M动态策略)</h2>
            <div class="plot-container">
                <h3>胜率热力图</h3>
                <img src="data:image/png;base64,{heatmap1_b64}" alt="Win Rate Heatmap">
            </div>
            <div class="plot-container">
                <h3>期望收益率热力图</h3>
                <img src="data:image/png;base64,{heatmap2_b64}" alt="Expected Return Heatmap">
            </div>
            <!-- ======================================================== -->
            {tables_2d_html} <!-- 新增的二维表格 -->
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
            # --- 新增：二维分析 (只针对M动态策略) ---
            m_dynamic_df = self._fetch_data_for_strategy(Q(one_stratage_mode__isnull=True))
            two_dim_analysis_results = self._analyze_2d_distribution(m_dynamic_df)
            
            # --- 新增：生成热力图 ---
            heatmap_win_rate_b64, heatmap_return_b64 = "", ""
            if two_dim_analysis_results:
                heatmap_win_rate_b64, heatmap_return_b64 = self._generate_heatmaps_base64(
                    two_dim_analysis_results['heatmap_win_rate'],
                    two_dim_analysis_results['heatmap_return']
                )
            html_content = self._format_html_content(
                all_analysis_results,
                plot1_b64,
                plot2_b64,
                two_dim_analysis_results,
                heatmap_win_rate_b64,
                heatmap_return_b64
            )
            
            subject = f"M值胜率分布回测报告 (多策略版) - {datetime.now().strftime('%Y-%m-%d')}"
            
            self.email_handler.send_email(
                recipients=self.recipients,
                subject=subject,
                html_content=html_content
            )
            logger.info(f"[{self.backtest_run_id}] 多策略回测报告邮件已成功发送。")
        except Exception as e:
            logger.error(f"[{self.backtest_run_id}] 生成或发送多策略报告时失败: {e}", exc_info=True)
