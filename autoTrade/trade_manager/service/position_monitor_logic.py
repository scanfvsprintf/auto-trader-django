# trade_manager/service/position_monitor_logic.py (新增或在monitor_exit_service.py中定义)
from decimal import Decimal
from common.models import Position, TradeLog

class PositionMonitorLogic:
    """
    持仓监控的核心决策逻辑。
    这是一个无状态的类，所有方法都是静态的，便于在任何地方调用。
    """
    @staticmethod
    def check_and_decide(position: Position, current_price: Decimal, params: dict) -> dict:
        """
        根据当前价格，对一个持仓做出决策。
        这是所有监控逻辑的唯一入口。

        :param position: 持仓对象
        :param current_price: 当前检查的价格
        :param params: 包含所需策略参数的字典
        :return: 一个包含决策的字典, e.g.,
                 {'action': 'SELL', 'reason': 'stop_loss', 'exit_price': Decimal('10.00')}
                 {'action': 'UPDATE', 'updates': {'current_stop_loss': ..., 'current_take_profit': ...}}
                 {'action': 'NONE'}
        """
        # 阶段一：止损退出检查 (最高优先级)
        if current_price <= position.current_stop_loss:
            final_reason = TradeLog.ReasonChoices.TAKE_PROFIT if position.current_stop_loss >= position.entry_price else TradeLog.ReasonChoices.STOP_LOSS
            return {
                'action': 'SELL',
                'reason': final_reason,
                'exit_price': position.current_stop_loss
            }

        # 阶段二：追踪止盈
        if current_price >= position.current_take_profit:
            new_tp = position.current_take_profit * (1 + params['trailing_tp_increment_pct'])
            new_sl = position.current_take_profit * (1 - params['trailing_sl_buffer_pct'])
            return {
                'action': 'UPDATE',
                'updates': {
                    'current_take_profit': new_tp.quantize(Decimal('0.01')),
                    'current_stop_loss': new_sl.quantize(Decimal('0.01'))
                }
            }

        # 阶段三：成本锁定
        if position.current_stop_loss < position.entry_price:
            base_price = max(position.entry_price, position.current_stop_loss)
            cost_lock_price = min(
                (base_price + position.current_take_profit) / 2,
                base_price * Decimal('1.01')
            )
            if current_price > cost_lock_price:
                new_sl = ((base_price + cost_lock_price) / 2)
                # 确保新的止损价不会高于当前价，避免立即触发
                if new_sl < current_price:
                    return {
                        'action': 'UPDATE',
                        'updates': {'current_stop_loss': new_sl.quantize(Decimal('0.01'))}
                    }

        return {'action': 'NONE'}
