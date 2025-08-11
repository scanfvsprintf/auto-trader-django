# trade_manager/service/decision_order_service.py

import logging
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import pandas_ta as ta

from django.db import transaction
from django.utils import timezone

from common.models import (
    DailyTradingPlan,
    Position,
    TradeLog,
    StrategyParameters,
    DailyQuotes,
    SystemLog
)
from .trade_handler import ITradeHandler

# 配置日志记录器
logger = logging.getLogger(__name__)

class DecisionOrderService:
    """
    开盘决策与下单模块。

    该服务负责在T日开盘后的黄金时间内，根据预案、实际开盘价和账户状态，
    做出最终的买入决策，并执行下单。同时，它也提供了在订单成交后计算
    止盈止损区间的功能。
    """
    MODULE_NAME = '开盘决策与下单'

    def __init__(self, handler: ITradeHandler, execution_date: date = None):
        """
        初始化服务。

        :param handler: 一个实现了 ITradeHandler 接口的实例，用于与交易环境交互。
        :param execution_date: T日，即执行决策的日期。如果为None，则默认为当天。
                               此参数为回测模块提供了设置模拟日期的入口。
        """
        if not isinstance(handler, ITradeHandler):
            raise TypeError("传入的 handler 必须是 ITradeHandler 的一个实例。")
        
        self.handler = handler
        self.execution_date = execution_date if execution_date else date.today()
        self.params = self._load_strategy_parameters()
        
        logger.debug(f"[{self.MODULE_NAME}] 服务初始化。执行T日: {self.execution_date}")
        logger.debug(f"策略参数加载成功: {self.params}")

    def _log_to_db(self, level: str, message: str):
        """辅助方法：将日志写入数据库"""
        SystemLog.objects.create(
            log_level=level,
            module_name=self.MODULE_NAME,
            message=message
        )

    def _load_strategy_parameters(self) -> dict:
        """从数据库加载所有策略参数到内存"""
        params = {}
        # 定义需要加载的参数及其默认值，以防数据库中没有
        required_params = {
            'MAX_POSITIONS': 2,
            'MAX_CAPITAL_PER_POSITION': 25000.00,
            'k_slip': 0.002,
            'Base_Target': 0.07,
            'k_g1': 1.5,
            'Max_Target': 0.20,
            'k_h1': 2.0,
            'k_h2': 3.0,
            'Max_Loss_Percent': 0.08,
            'lookback_atr': 14,
            'lookback_adx': 14,
            'lookback_ma20': 20,
            'param_adx_threshold': 25
        }
        
        db_params = {p.param_name: p.param_value for p in StrategyParameters.objects.all()}
        
        for key, default_value in required_params.items():
            # 优先使用数据库中的值，否则使用默认值
            value = db_params.get(key, Decimal(str(default_value)))
            # 将需要整数的参数转换为int
            if key in ['MAX_POSITIONS', 'lookback_atr', 'lookback_adx', 'lookback_ma20', 'param_adx_threshold']:
                params[key] = int(value)
            else:
                params[key] = Decimal(str(value))
        return params

    # --- 暴露给外部调度的核心函数 ---

    def adjust_trading_plan_daily(self):
        """
        函数一：执行每日交易预案再调整。
        根据实际开盘价与剩余仓位进行二次筛选，关闭不会被选择的交易预案。
        """
        logger.debug(f"开始执行 {self.execution_date} 的交易预案二次筛选...")
        
        plans_today = DailyTradingPlan.objects.filter(
            plan_date=self.execution_date,
            status=DailyTradingPlan.StatusChoices.PENDING
        ).order_by('rank')

        if not plans_today.exists():
            msg = f"在 {self.execution_date} 没有找到待执行的交易预案。"
            logger.warning(msg)
            self._log_to_db('WARNING', msg)
            return

        plans_to_cancel = []
        for plan in plans_today:
            try:
                open_price = self.handler.get_opening_price(plan.stock_code)
                if open_price <= 0:
                    logger.warning(f"股票 {plan.stock_code} 开盘价为0或无效，视为不符合条件。")
                    plan.status = DailyTradingPlan.StatusChoices.CANCELLED
                    plans_to_cancel.append(plan)
                    continue

                if not (plan.miop <= open_price <= plan.maop):
                    msg = (f"预案 {plan.stock_code} (Rank:{plan.rank}) 开盘价 {open_price} "
                           f"不在区间 [{plan.miop}, {plan.maop}] 内，已作废。")
                    logger.debug(msg)
                    plan.status = DailyTradingPlan.StatusChoices.CANCELLED
                    plans_to_cancel.append(plan)

            except Exception as e:
                msg = f"获取 {plan.stock_code} 开盘价时发生错误: {e}，该预案作废。"
                logger.error(msg)
                self._log_to_db('ERROR', msg)
                plan.status = DailyTradingPlan.StatusChoices.CANCELLED
                plans_to_cancel.append(plan)

        if plans_to_cancel:
            with transaction.atomic():
                DailyTradingPlan.objects.bulk_update(plans_to_cancel, ['status'])
            logger.info(f"成功作废 {len(plans_to_cancel)} 条不符合开盘条件的交易预案。")
        else:
            logger.info("所有待执行预案均符合开盘价条件。")

    def execute_orders(self):
        """
        函数二：进行下单。
        读取预案表，选择最优标的，计算仓位和价格，并调用处理器执行下单。
        """
        logger.info(f"开始执行 {self.execution_date} 的下单流程...")

        # 1. 检查剩余仓位
        open_positions_count = Position.objects.filter(status=Position.StatusChoices.OPEN).count()
        remaining_slots = self.params['MAX_POSITIONS'] - open_positions_count

        if remaining_slots <= 0:
            msg = f"当前持仓数 {open_positions_count} 已达上限 {self.params['MAX_POSITIONS']}，今日不进行买入操作。"
            logger.warning(msg)
            self._log_to_db('WARNING', msg)
            return

        #2. 获取所有待处理的候选标的
        candidates = DailyTradingPlan.objects.filter(
            plan_date=self.execution_date,
            status=DailyTradingPlan.StatusChoices.PENDING
        ).order_by('rank')
 
        if not candidates.exists():
            msg = f"在 {self.execution_date} 无符合条件的买入标的。"
            logger.info(msg)
            self._log_to_db('INFO', msg)
            return

        # 3. 遍历所有候选标的，直到成功买入一个
        for candidate in candidates:
            try:
                stock_code = candidate.stock_code
                open_price = self.handler.get_opening_price(stock_code)
                
                # 计算下单限价
                k_slip = self.params['k_slip']
                limit_price = (open_price * (Decimal('1.0') + k_slip)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
 
                # 计算本次交易可用资金
                available_balance = self.handler.get_available_balance()
                capital_per_slot = available_balance / Decimal(remaining_slots)
                nominal_principal = min(capital_per_slot, self.params['MAX_CAPITAL_PER_POSITION'])
 
                # 计算购入股数
                if limit_price <= 0:
                    logger.warning(f"标的 {stock_code}: 计算出的下单限价无效（{limit_price}），跳过。")
                    continue # 尝试下一个候选
 
                shares_to_buy = int(nominal_principal / limit_price)
                quantity = (shares_to_buy // 100) * 100 # 向下取整到100的倍数
 
                if quantity < 100:
                    msg = (f"标的 {stock_code}: 计算出的名义本金 {nominal_principal:.2f} 不足以购买一手（100股），"
                           f"所需金额约为 {limit_price * 100:.2f}。放弃本次交易。")
                    logger.warning(msg)
                    self._log_to_db('WARNING', msg)
                    continue # 资金不足，尝试下一个候选
 
                # 4. 执行下单
                msg = (f"确定唯一买入标的: {stock_code} (Rank:{candidate.rank})。 "
                       f"计划以限价 {limit_price} 买入 {quantity} 股。")
                logger.info(msg)
                self._log_to_db('INFO', msg)
                
                self.handler.place_buy_order(stock_code, limit_price, quantity)
                
                # 标记预案为已执行
                candidate.status = DailyTradingPlan.StatusChoices.EXECUTED
                candidate.save()
 
                # 成功买入后，立即退出函数，外层循环会决定是否继续买入下一个仓位
                return
 
            except Exception as e:
                msg = f"处理候选股 {candidate.stock_code} 时发生严重错误: {e}"
                logger.error(msg, exc_info=True)
                self._log_to_db('CRITICAL', msg)
                continue # 发生异常，继续尝试下一个候选
 
        # 如果循环正常结束，说明所有候选股都无法买入
        logger.info(f"已尝试所有 {len(candidates)} 个候选标的，均未成功买入。")

    def calculate_stop_profit_loss(self, trade_id: int):
        """
        函数三：止盈止损区间计算 (修正版)。
        在订单成交后，为新持仓计算并更新初始的止盈止损价。
 
        :param trade_id: 已成交的买入交易在 tb_trade_log 中的唯一ID。
        """
        logger.info(f"开始为 trade_id={trade_id} 计算止盈止损区间...")
        try:
            with transaction.atomic():
                # 1. 获取交易和持仓信息
                trade_log = TradeLog.objects.select_for_update().get(
                    trade_id=trade_id,
                    trade_type=TradeLog.TradeTypeChoices.BUY,
                    status=TradeLog.StatusChoices.FILLED
                )
                position = Position.objects.select_for_update().get(pk=trade_log.position_id)
 
                if position.current_stop_loss > 0:
                    logger.warning(f"Position ID {position.position_id} 似乎已计算过止盈止损，将跳过。")
                    return
 
                stock_code = trade_log.stock_code_id
                aep = trade_log.price
                buy_date = trade_log.trade_datetime.date()
                
                # 2. 获取计算所需行情数据 (避免未来函数)
                lookback_days = self.params['lookback_adx'] + 50
                start_date_for_calc = buy_date - timedelta(days=lookback_days * 2)
                end_date_for_calc = buy_date - timedelta(days=1)
 
                quotes_qs = DailyQuotes.objects.filter(
                    stock_code_id=stock_code,
                    trade_date__gte=start_date_for_calc,
                    trade_date__lte=end_date_for_calc
                ).order_by('trade_date')
 
                if len(quotes_qs) < max(self.params['lookback_atr'], self.params['lookback_adx'], self.params['lookback_ma20']):
                    raise ValueError(f"股票 {stock_code} 在 {end_date_for_calc} 前的历史数据不足，无法计算指标。")
 
                df = pd.DataFrame.from_records(quotes_qs.values('high', 'low', 'close'))
                df = df.astype(float)
 
                # 3. 计算所有必需指标
                atr_series = ta.atr(df['high'], df['low'], df['close'], length=self.params['lookback_atr'])
                atr_14_buy = Decimal(str(atr_series.iloc[-1]))
 
                ma20_series = ta.sma(df['close'], length=self.params['lookback_ma20'])
                ma20_buy = Decimal(str(ma20_series.iloc[-1]))
 
                adx_df = ta.adx(df['high'], df['low'], df['close'], length=self.params['lookback_adx'])
                adx_14_buy = Decimal(str(adx_df[f'ADX_{self.params["lookback_adx"]}'].iloc[-1]))
 
                # 4. 计算止盈价 g(y) - 逻辑不变
                profit_margin = min(
                    self.params['Base_Target'] + self.params['k_g1'] * (atr_14_buy / aep),
                    self.params['Max_Target']
                )
                take_profit_price = aep * (Decimal('1.0') + profit_margin)
 
                # 5. 计算止损价 h(z) - 严格按照需求文档逻辑
                # 5.1 根据ADX判断市场状态，选择z_final
                adx_threshold = self.params['param_adx_threshold']
                if adx_14_buy > adx_threshold:
                    # 趋势状态，使用较窄的ATR乘数
                    z_final = aep - self.params['k_h1'] * atr_14_buy
                else:
                    # 震荡状态，使用较宽的ATR乘数
                    z_final = aep - self.params['k_h2'] * atr_14_buy
 
                # 5.2 计算其他止损线
                z2_technical = ma20_buy
                z3_max_loss = aep * (Decimal('1.0') - self.params['Max_Loss_Percent'])
                
                # 5.3 取最严格的止损位（价格最高者）
                stop_loss_price = max(z_final, z2_technical, z3_max_loss)
                
                logger.info(f"[{stock_code}] 止损线比较: 趋势位={z_final:.2f}, 技术位={z2_technical:.2f}, 底线={z3_max_loss:.2f}")
 
                # 6. 更新持仓信息表
                position.current_take_profit = take_profit_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                position.current_stop_loss = stop_loss_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                position.save(update_fields=['current_take_profit', 'current_stop_loss'])
 
                msg = (f"成功计算并更新 Position ID {position.position_id} ({stock_code}) 的风控价格: "
                       f"购入价={aep}, 止盈价={position.current_take_profit}, 止盈率={((Decimal('1.0') + profit_margin)*100):.2f}%, 止损价={position.current_stop_loss}, 止损率={((position.current_stop_loss/aep)*100):.2f}%")
                logger.info(msg)
                self._log_to_db('INFO', msg)
 
        except TradeLog.DoesNotExist:
            logger.error(f"Trade ID {trade_id} 不存在或不满足计算条件（非买入/未成交）。")
        except Position.DoesNotExist:
            logger.error(f"与 Trade ID {trade_id} 关联的 Position 不存在。")
        except Exception as e:
            msg = f"为 Trade ID {trade_id} 计算止盈止损时发生严重错误: {e}"

    # --- 工具函数 ---

    @staticmethod
    def initialize_strategy_parameters():
        """
        工具函数：初始化本模块所需的策略参数到数据库。
        这是一个幂等操作，可以安全地重复运行。
        """
        logger.info("开始初始化[开盘决策与下单模块]的策略参数...")

        params_to_define = {
            # 仓位管理
            'MAX_POSITIONS': {'value': '2', 'group': 'POSITION_MGMT', 'desc': '最大可具备的总仓位数'},
            'MAX_CAPITAL_PER_POSITION': {'value': '25000.00', 'group': 'POSITION_MGMT', 'desc': '每仓最大投入资金数(元)'},
            # 下单参数
            'k_slip': {'value': '0.002', 'group': 'ORDER_EXEC', 'desc': '下单滑点系数, 用于计算限价单价格'},
            # 止盈参数 g(y)
            'Base_Target': {'value': '0.07', 'group': 'TAKE_PROFIT', 'desc': '基础止盈目标百分比'},
            'k_g1': {'value': '1.5', 'group': 'TAKE_PROFIT', 'desc': 'ATR溢价乘数, 用于动态调整止盈目标'},
            'Max_Target': {'value': '0.20', 'group': 'TAKE_PROFIT', 'desc': '最大止盈目标百分比上限'},
            # 止损参数 h(z)
            'k_h1': {'value': '2.0', 'group': 'STOP_LOSS', 'desc': '趋势市ATR止损乘数 (盘中动态使用)'},
            'k_h2': {'value': '3.0', 'group': 'STOP_LOSS', 'desc': '震荡市ATR止损乘数 (用于计算初始止损)'},
            'Max_Loss_Percent': {'value': '0.08', 'group': 'STOP_LOSS', 'desc': '最大回撤容忍度(绝对亏损百分比上限)'},
            # 指标周期
            'lookback_atr': {'value': '14', 'group': 'INDICATORS', 'desc': 'ATR计算周期'},
            'lookback_adx': {'value': '14', 'group': 'INDICATORS', 'desc': 'ADX计算周期'},
            'lookback_ma20': {'value': '20', 'group': 'INDICATORS', 'desc': 'MA20计算周期'},
        }

        with transaction.atomic():
            for name, data in params_to_define.items():
                StrategyParameters.objects.update_or_create(
                    param_name=name,
                    defaults={
                        'param_value': Decimal(data['value']),
                        'group_name': data['group'],
                        'description': data['desc']
                    }
                )
        
        logger.info(f"成功初始化/更新 {len(params_to_define)} 个策略参数。")

