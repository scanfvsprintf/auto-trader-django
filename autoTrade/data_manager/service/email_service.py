# data_manager/service/email_service.py

import logging
from datetime import date, timedelta
from decimal import Decimal

from django.utils import timezone
from django.db import transaction

# 内部模块导入
from .email_handler import EmailHandler

# Django模型导入
from common.models import (
    DailyFactorValues, DailyTradingPlan, DailyQuotes, StockInfo,
    Position, TradeLog
)
from selection_manager.service.selection_service import MARKET_INDICATOR_CODE
from trade_manager.service.decision_order_service import DecisionOrderService
from trade_manager.service.simulate_trade_handler import SimulateTradeHandler
from trade_manager.service.simulate_trade import SimulateTradeService


logger = logging.getLogger(__name__)

class EmailNotificationService:
    """
    封装了在T日盘前（如9:10）向指定邮箱推送T-1日预案的业务逻辑。
    """

    def __init__(self, t_day: date):
        """
        初始化邮件通知服务。
        :param t_day: T日，即预案执行日。
        """
        self.t_day = t_day
        try:
            # 获取T日之前的最后一个交易日作为T-1日
            self.t_minus_1_day = DailyQuotes.objects.filter(
                trade_date__lt=self.t_day
            ).latest('trade_date').trade_date
        except DailyQuotes.DoesNotExist:
            raise ValueError(f"无法找到 {self.t_day} 的前一个交易日(T-1)，服务无法初始化。")

        self.email_handler = EmailHandler()
        # 从EmailHandler的配置中直接读取收件人列表
        self.recipients = self.email_handler.recipients if hasattr(self.email_handler, 'recipients') else ['876858298@qq.com','850696281@qq.com','285173686@qq.com','2516937525@qq.com']


    def runEmailSend(self):
        """
        一键执行邮件发送的主方法。
        """
        logger.info(f"开始为T日({self.t_day})生成预案推送邮件...")
        # 【核心修改】先找到真正需要处理的预案日期
        plan_date_to_process = self._find_latest_pending_plan_date()
        if not plan_date_to_process:
            logger.warning(f"从T日({self.t_day})回溯，未找到任何待处理的交易预案，邮件发送任务终止。")
            return
        # 1. 获取所有需要的数据
        market_data = self._get_market_regime_data()
        # 【核心修改】将找到的日期传递给下一步
        plan_details = self._get_trading_plan_details(plan_date_to_process)
        if not plan_details:
            logger.warning(f"预案日({plan_date_to_process})的预案详情为空，邮件发送任务终止。")
            return
        # 2. 生成HTML内容
        html_content = self._format_html_content(market_data, plan_details)
        # 3. 发送邮件
        subject = f"【交易预案】{self.t_day.strftime('%Y-%m-%d')} 盘前确认 (数据源: {plan_date_to_process.strftime('%Y-%m-%d')})"
        self.email_handler.send_email(self.recipients, subject, html_content)

    def _get_market_regime_data(self) -> dict:
        """获取昨日M值及近10日M值历史"""
        try:
            # 获取T-1日及之前的10个交易日
            trade_dates = list(DailyQuotes.objects.filter(trade_date__lte=self.t_minus_1_day)
                               .values_list('trade_date', flat=True)
                               .distinct().order_by('-trade_date')[:10])
            trade_dates.reverse()

            m_values_qs = DailyFactorValues.objects.filter(
                stock_code_id=MARKET_INDICATOR_CODE,
                factor_code_id='dynamic_M_VALUE',
                trade_date__in=trade_dates
            ).order_by('-trade_date')

            m_values_map = {fv.trade_date: fv.raw_value for fv in m_values_qs}

            yesterday_m = m_values_map.get(self.t_minus_1_day, Decimal('NaN'))
            history_m = [{'date': d, 'value': m_values_map.get(d, Decimal('NaN'))} for d in trade_dates]

            return {'yesterday_m': yesterday_m, 'history_m': history_m}

        except Exception as e:
            logger.error(f"获取M值数据时出错: {e}", exc_info=True)
            return {'yesterday_m': Decimal('NaN'), 'history_m': []}

    def _get_trading_plan_details(self, plan_date: date) -> list[dict]:
        """获取T日交易预案及相关的所有详细信息"""
        plans = DailyTradingPlan.objects.filter(plan_date=plan_date, status=DailyTradingPlan.StatusChoices.PENDING).order_by('rank')
        if not plans.exists():
            return []

        detailed_plans = []
        for plan in plans:
            stock_code = plan.stock_code_id
            logger.debug(f"正在处理预案股票: {stock_code}")
            try:
                # 获取止盈止损率
                rates = self._calculate_profit_loss_rates(stock_code)
                # 获取历史行情
                history = self._get_stock_historical_data(stock_code)

                detailed_plans.append({
                    'plan': plan,
                    'stock_info': plan.stock_code, # StockInfo object
                    'rates': rates,
                    'history': history
                })
            except Exception as e:
                logger.error(f"处理股票 {stock_code} 的预案详情时失败: {e}", exc_info=True)
                continue # 跳过这个出错的股票

        return detailed_plans

    def _calculate_profit_loss_rates(self, stock_code: str) -> dict:
        """
        通过创建临时数据库记录来复用现有止盈止损计算逻辑。
        整个过程在单个数据库事务中完成，确保安全。
        """
        tp_rate, sl_rate = Decimal('NaN'), Decimal('NaN')
        try:
            # 【新增步骤1】: 获取T-1日收盘价作为基准
            try:
                t_minus_1_quote = DailyQuotes.objects.get(stock_code_id=stock_code, trade_date=self.t_minus_1_day)
                base_price = t_minus_1_quote.close
                if base_price <= 0:
                    raise ValueError("T-1日收盘价无效")
            except DailyQuotes.DoesNotExist:
                logger.error(f"无法找到 {stock_code} 在 {self.t_minus_1_day} 的行情数据，无法计算止盈止损率。")
                return {'tp_rate': tp_rate, 'sl_rate': sl_rate}
            except ValueError as e:
                logger.error(f"股票 {stock_code} 在 {self.t_minus_1_day} 的收盘价不合法: {e}")
                return {'tp_rate': tp_rate, 'sl_rate': sl_rate}
            with transaction.atomic():
                # 【修改步骤2】: 使用获取到的base_price创建临时记录
                temp_position = Position.objects.create(
                    stock_code_id=stock_code,
                    entry_price=base_price, # 使用T-1收盘价
                    quantity=100,
                    entry_datetime=timezone.now(),
                    status=Position.StatusChoices.OPEN,
                    current_stop_loss=Decimal('0.00'),
                    current_take_profit=Decimal('0.00')
                )
                temp_trade_log = TradeLog.objects.create(
                    position=temp_position,
                    stock_code_id=stock_code,
                    trade_datetime=timezone.now(),
                    trade_type=TradeLog.TradeTypeChoices.BUY,
                    status=TradeLog.StatusChoices.FILLED,
                    price=base_price, # 使用T-1收盘价
                    quantity=100,
                    commission=0,
                    stamp_duty=0
                )
                # 调用服务进行计算 (这部分不变)
                dummy_sim_service = SimulateTradeService()
                dummy_handler = SimulateTradeHandler(dummy_sim_service)
                decision_service = DecisionOrderService(handler=dummy_handler, execution_date=self.t_day)
                decision_service.calculate_stop_profit_loss(trade_id=temp_trade_log.trade_id)
                temp_position.refresh_from_db()
                # 【修改步骤3】: 使用base_price作为分母计算比率
                take_profit_price = temp_position.current_take_profit
                stop_loss_price = temp_position.current_stop_loss
                if take_profit_price > 0:
                    tp_rate = (take_profit_price / base_price) - 1
                if stop_loss_price > 0:
                    sl_rate = 1 - (stop_loss_price / base_price)
                # 回滚事务，清除临时数据 (这部分不变)
                transaction.set_rollback(True)
        except Exception as e:
            logger.error(f"为 {stock_code} 计算止盈止损率时发生严重错误: {e}", exc_info=True)
            transaction.set_rollback(True)
        logger.debug(f"{stock_code} (基准价: {base_price:.2f}) -> TP Rate: {tp_rate:.4%}, SL Rate: {sl_rate:.4%}")
        return {'tp_rate': tp_rate, 'sl_rate': sl_rate}


    def _get_stock_historical_data(self, stock_code: str) -> list[dict]:
        """获取指定股票近10个交易日的历史行情"""
        trade_dates = list(DailyQuotes.objects.filter(trade_date__lte=self.t_minus_1_day)
                           .values_list('trade_date', flat=True)
                           .distinct().order_by('-trade_date')[:10])
        trade_dates.reverse()

        quotes = DailyQuotes.objects.filter(
            stock_code_id=stock_code,
            trade_date__in=trade_dates
        ).order_by('trade_date')

        history = []
        prev_close = None
        for quote in quotes:
            change_pct = Decimal('0.0')
            if prev_close and prev_close > 0:
                change_pct = (quote.close / prev_close) - 1
            
            history.append({
                'date': quote.trade_date,
                'open': quote.open,
                'high': quote.high,
                'low': quote.low,
                'close': quote.close,
                'hfq_close': quote.hfq_close,
                'change_pct': change_pct
            })
            prev_close = quote.close
        return history

    def _format_html_content(self, market_data: dict, plan_details: list[dict]) -> str:
        """将所有数据格式化为美观的HTML字符串"""
        
        # --- CSS样式 ---
        style = """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa; margin: 0; padding: 20px; }
            .container { max-width: 800px; margin: auto; background: #fff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
            h2 { color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; margin-top: 30px; }
            h3 { color: #17a2b8; margin-top: 25px; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }
            th, td { border: 1px solid #dee2e6; padding: 10px; text-align: left; }
            th { background-color: #e9ecef; font-weight: 600; }
            tr:nth-child(even) { background-color: #f8f9fa; }
            .summary { font-size: 16px; font-weight: bold; margin-bottom: 20px; }
            .red { color: #dc3545; }
            .green { color: #28a745; }
            .footer { margin-top: 30px; font-size: 12px; color: #6c757d; text-align: center; }
        </style>
        """

        # --- HTML头部 ---
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>交易预案确认</title>
            {style}
        </head>
        <body>
            <div class="container">
                <h2>T日 ({self.t_day.strftime('%Y-%m-%d')}) 交易预案盘前确认</h2>
        """

        # --- 大盘情况 ---
        yesterday_m_str = f"{market_data['yesterday_m']:.4f}" if not market_data['yesterday_m'].is_nan() else "N/A"
        html += f"""
        <h3>[大盘情况]</h3>
        <p class="summary">昨日M值: <span class="{'red' if market_data.get('yesterday_m', 0) > 0 else 'green'}">{yesterday_m_str}</span></p>
        <table>
            <thead><tr><th>日期</th><th>M值</th></tr></thead>
            <tbody>
        """
        for item in reversed(market_data['history_m']):
            m_val_str = f"{item['value']:.4f}" if not item['value'].is_nan() else "N/A"
            html += f"<tr><td>{item['date'].strftime('%Y-%m-%d')}</td><td>{m_val_str}</td></tr>"
        html += "</tbody></table>"

        # --- 选股预案 ---
        html += "<h3>[选股预案]</h3>"
        html += """
        <table>
            <thead>
                <tr>
                    <th>排名</th>
                    <th>股票代码</th>
                    <th>股票名称</th>
                    <th>可接受开盘区间</th>
                    <th>预期止盈率</th>
                    <th>预期止损率</th>
                </tr>
            </thead>
            <tbody>
        """
        for detail in plan_details:
            plan = detail['plan']
            stock_info = detail['stock_info']
            rates = detail['rates']
            tp_rate_str = f"{rates['tp_rate']:.2%}" if not rates['tp_rate'].is_nan() else "N/A"
            sl_rate_str = f"{rates['sl_rate']:.2%}" if not rates['sl_rate'].is_nan() else "N/A"
            html += f"""
            <tr>
                <td>{plan.rank}</td>
                <td>{stock_info.stock_code}</td>
                <td>{stock_info.stock_name}</td>
                <td>{plan.miop:.2f} - {plan.maop:.2f}</td>
                <td class="red">{tp_rate_str}</td>
                <td class="green">{sl_rate_str}</td>
            </tr>
            """
        html += "</tbody></table>"

        # --- 各股票历史行情 ---
        for detail in plan_details:
            stock_info = detail['stock_info']
            history = detail['history']
            html += f"<h4>{stock_info.stock_name} ({stock_info.stock_code}) - 近10日行情</h4>"
            html += """
            <table>
                <thead>
                    <tr>
                        <th>日期</th>
                        <th>开盘价</th>
                        <th>最高价</th>
                        <th>最低价</th>
                        <th>收盘价</th>
                        <th>后复权收盘</th>
                        <th>涨幅</th>
                    </tr>
                </thead>
                <tbody>
            """
            for item in reversed(history):
                color_class = 'red' if item['change_pct'] > 0 else ('green' if item['change_pct'] < 0 else '')
                html += f"""
                <tr>
                    <td>{item['date'].strftime('%Y-%m-%d')}</td>
                    <td>{item['open']:.2f}</td>
                    <td>{item['high']:.2f}</td>
                    <td>{item['low']:.2f}</td>
                    <td>{item['close']:.2f}</td>
                    <td>{item['hfq_close']:.4f}</td>
                    <td class="{color_class}">{item['change_pct']:.2%}</td>
                </tr>
                """
            html += "</tbody></table>"

        # --- HTML尾部 ---
        html += """
                <p class="footer">本邮件由策略交易系统自动生成，仅供参考，请在交易前最终确认。</p>
            </div>
        </body>
        </html>
        """
        return html
    def _find_latest_pending_plan_date(self) -> date | None:
        """从T日开始向前回溯，查找最新的一个包含待执行预案的日期"""
        # 设置一个合理的回溯上限，例如14天
        for i in range(14):
            check_date = self.t_day - timedelta(days=i)
            if DailyTradingPlan.objects.filter(
                plan_date=check_date,
                status=DailyTradingPlan.StatusChoices.PENDING
            ).exists():
                logger.info(f"找到待执行的交易预案，预案生成日为: {check_date}")
                return check_date
        logger.warning(f"在过去14天内未找到任何待执行的交易预案。")
        return None
