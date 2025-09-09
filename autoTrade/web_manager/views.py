from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.db import models
from django.db import connection
from django.db.models import F
from datetime import datetime, timedelta
from decimal import Decimal

from common.models import (
    DailyTradingPlan, StockInfo, DailyFactorValues, FactorDefinitions,
    DailyQuotes, IndexQuotesCsi300, StrategyParameters
)
from data_manager.service.stock_service import StockService
from selection_manager.service.selection_service import SelectionService


def ok(data=None):
    return JsonResponse({"code": 0, "msg": "ok", "data": data}, safe=False)

def err(msg, code=1):
    return JsonResponse({"code": code, "msg": msg, "data": None}, status=200)
# ----------------------- 通用：股票搜索 -----------------------
@require_http_methods(["GET"])
def stock_search(request):
    """
    模糊搜索股票：支持代码或名称 contains。
    参数: q  结果最多返回 20 条
    """
    q = request.GET.get('q', '').strip()
    if not q:
        return ok([])
    qs = StockInfo.objects.filter(models.Q(stock_code__icontains=q) | models.Q(stock_name__icontains=q)).values('stock_code', 'stock_name')[:20]
    return ok(list(qs))


# ----------------------- 选股管理 -----------------------
@require_http_methods(["GET"])
def selection_plans(request):
    """
    查询选股预案：
    - 输入 date（YYYY-MM-DD）。若当日无预案，则在 [date-30天, date] 内回溯最近一个有预案的交易日并返回。
    - 输出结构统一为 { date: 实际返回日期, rows: [...] }
    """
    plan_date_str = request.GET.get('date')
    if not plan_date_str:
        return err("缺少参数: date")
    try:
        plan_date = datetime.strptime(plan_date_str, '%Y-%m-%d').date()
    except Exception:
        return err("参数 date 格式应为 YYYY-MM-DD")

    # 首先尝试当日
    base_qs = DailyTradingPlan.objects.filter(plan_date=plan_date)
    actual_date = plan_date
    if not base_qs.exists():
        # 回溯最近交易日，最多30天
        start_date = plan_date - timedelta(days=30)
        nearest = (
            DailyTradingPlan.objects
            .filter(plan_date__lte=plan_date, plan_date__gte=start_date)
            .order_by('-plan_date')
            .values_list('plan_date', flat=True)
            .first()
        )
        if nearest:
            actual_date = nearest
            base_qs = DailyTradingPlan.objects.filter(plan_date=actual_date)
        else:
            return ok({"date": plan_date_str, "rows": []})

    qs = (
        base_qs
        .select_related('stock_code')
        .order_by('rank')
        .values(
            'rank', 'miop', 'maop', 'final_score',
            'stock_code_id',
            stock_name=F('stock_code__stock_name')
        )
    )
    rows = []
    for row in qs:
        rows.append({
            'rank': row['rank'],
            'miop': row['miop'],
            'maop': row['maop'],
            'final_score': row['final_score'],
            'stock_code': row['stock_code_id'],
            'stock_name': row['stock_name'],
        })
    return ok({"date": actual_date.isoformat(), "rows": rows})


@require_http_methods(["GET"])
def selection_factors(request):
    date_str = request.GET.get('date')
    stock_code = request.GET.get('stock')
    if not date_str or not stock_code:
        return err("缺少参数: date 或 stock")
    try:
        plan_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except Exception:
        return err("参数 date 格式应为 YYYY-MM-DD")

    # 因子取上一个交易日（严格小于计划日的最近交易日）
    prev_trade_date = (
        DailyQuotes.objects
        .filter(trade_date__lt=plan_date)
        .order_by('-trade_date')
        .values_list('trade_date', flat=True)
        .first()
    )
    if not prev_trade_date:
        return ok([])

    qs = (
        DailyFactorValues.objects
        .filter(stock_code_id=stock_code, trade_date=prev_trade_date)
        .select_related('factor_code')
        .values(
            'raw_value', 'norm_score',
            'factor_code_id',
            factor_name=F('factor_code__factor_name')
        ).order_by('factor_code_id')
    )
    data = []
    for row in qs:
        raw_val = row['raw_value']
        norm_val = row['norm_score']
        try:
            raw_val = float(raw_val) if raw_val is not None else None
        except Exception:
            pass
        try:
            norm_val = float(norm_val) if norm_val is not None else None
        except Exception:
            pass
        data.append({
            'factor_code': row['factor_code_id'],
            'factor_name': row['factor_name'] or row['factor_code_id'],
            'raw_value': raw_val,
            'norm_score': norm_val,
        })
    return ok(data)


@require_http_methods(["POST"])
def selection_run(request):
    try:
        import json
        payload = json.loads(request.body.decode('utf-8') or '{}')
    except Exception:
        payload = {}
    date_str = payload.get('date')
    send_mail = bool(payload.get('send_mail', False))
    if not date_str:
        return err("缺少参数: date")
    try:
        t1 = datetime.strptime(date_str, '%Y-%m-%d').date()
    except Exception:
        return err("参数 date 格式应为 YYYY-MM-DD")

    svc = SelectionService(trade_date=t1, mode='realtime')
    svc.run_selection()
    return ok({"executed": True, "send_mail": send_mail})


@require_http_methods(["POST"])
def selection_run_range(request):
    """
    在日期区间内回补选股/评分：
    - 参数：start, end（YYYY-MM-DD）
    - 逻辑：
      1) 找出区间内的交易日（来自 tb_daily_quotes 的 distinct trade_date）。
      2) 预加载滚动窗口所需的行情面板（open/high/low/close/volume/turnover）。
      3) 对每个交易日，构造 SelectionService(trade_date=该日, mode='backtest', preloaded_panels=窗口切片)，仅执行选股与保存结果；
         注意 SelectionService 内部会将 ML_STOCK_SCORE 存为当日，并把交易预案写入 T+1。
    """
    try:
        import json
        payload = json.loads(request.body.decode('utf-8') or '{}')
    except Exception:
        payload = {}
    start = payload.get('start')
    end = payload.get('end')
    if not start or not end:
        return err("缺少参数: start 或 end")

    try:
        from datetime import date, timedelta
        s = datetime.strptime(start, '%Y-%m-%d').date()
        e = datetime.strptime(end, '%Y-%m-%d').date()
    except Exception:
        return err("参数 start/end 格式应为 YYYY-MM-DD")

    if s > e:
        s, e = e, s

    # 1) 找出该区间内的交易日
    trading_days_qs = (
        DailyQuotes.objects
        .filter(trade_date__gte=s, trade_date__lte=e)
        .order_by('trade_date')
        .values_list('trade_date', flat=True)
        .distinct()
    )
    trading_days = list(trading_days_qs)
    if not trading_days:
        return ok({"executed": False, "msg": "区间内无交易日"})

    # 2) 预加载滚动窗口数据（一次查全量，后续按天切片），向量化效率更高
    lookback_window_size = 250
    extra_buffer_days = 20
    preload_start = trading_days[0] - timedelta(days=lookback_window_size + extra_buffer_days)

    quotes_qs = DailyQuotes.objects.filter(
        trade_date__gte=preload_start,
        trade_date__lte=trading_days[-1]
    ).values('trade_date', 'stock_code_id', 'open', 'high', 'low', 'close', 'volume', 'turnover')

    import pandas as pd
    df = pd.DataFrame.from_records(quotes_qs)
    if df.empty:
        return ok({"executed": False, "msg": "预加载行情为空"})
    # 转数值
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 构建全量面板
    panels_all = {}
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        panel = df.pivot(index='trade_date', columns='stock_code_id', values=col)
        panels_all[col] = panel

    executed_days = 0
    for td in trading_days:
        # 3) 为当前交易日构造滚动窗口切片（按 index 过滤日期）
        window_start = td - timedelta(days=lookback_window_size)
        sliced = {}
        for k, panel in panels_all.items():
            try:
                sliced[k] = panel.loc[(panel.index >= window_start) & (panel.index <= td)]
            except Exception:
                sliced[k] = panel
        try:
            svc = SelectionService(trade_date=td, mode='backtest', preloaded_panels=sliced)
            svc.run_selection()
            executed_days += 1
        except Exception as ex:
            # 不中断整体流程
            continue

    return ok({"executed": True, "days": executed_days})


# ----------------------- 日线管理 -----------------------
@require_http_methods(["GET"])
def daily_csi300(request):
    start = request.GET.get('start')
    end = request.GET.get('end')
    with_m = request.GET.get('with_m', '0') == '1'
    if not start or not end:
        return err("缺少参数: start 或 end")
    try:
        s = datetime.strptime(start, '%Y-%m-%d').date()
        e = datetime.strptime(end, '%Y-%m-%d').date()
    except Exception:
        return err("参数 start/end 格式应为 YYYY-MM-DD")
    qs = IndexQuotesCsi300.objects.filter(trade_date__gte=s, trade_date__lte=e).order_by('trade_date')
    data = list(qs.values('trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount'))
    if with_m:
        m_qs = DailyFactorValues.objects.filter(
            stock_code_id='_MARKET_REGIME_INDICATOR_',
            factor_code_id='dynamic_M_VALUE',
            trade_date__gte=s, trade_date__lte=e
        ).order_by('trade_date').values('trade_date', 'raw_value')
        m_map = {x['trade_date']: x['raw_value'] for x in m_qs}
        for row in data:
            row['m_value'] = m_map.get(row['trade_date'])
    return ok(data)


@require_http_methods(["POST"])
def daily_csi300_fetch(request):
    try:
        import json
        payload = json.loads(request.body.decode('utf-8') or '{}')
    except Exception:
        payload = {}
    start = payload.get('start')
    end = payload.get('end')
    if not start or not end:
        return err("缺少参数: start 或 end")
    svc = StockService()
    svc.update_csi300_index_data(start, end)
    return ok({"fetched": True})


@require_http_methods(["GET"])
def daily_stock(request):
    code = request.GET.get('code')
    start = request.GET.get('start')
    end = request.GET.get('end')
    ma = request.GET.get('ma', '')
    with_vol = request.GET.get('with_vol', '0') == '1'
    with_amt = request.GET.get('with_amt', '0') == '1'
    with_m = request.GET.get('with_m', '0') == '1'
    hfq = request.GET.get('hfq', '0') == '1'
    if not code or not start or not end:
        return err("缺少参数: code/start/end")
    try:
        s = datetime.strptime(start, '%Y-%m-%d').date()
        e = datetime.strptime(end, '%Y-%m-%d').date()
    except Exception:
        return err("参数 start/end 格式应为 YYYY-MM-DD")
    qs = DailyQuotes.objects.filter(stock_code_id=code, trade_date__gte=s, trade_date__lte=e).order_by('trade_date')
    fields = ['trade_date', 'open', 'high', 'low', 'close', 'adjust_factor']
    if with_vol:
        fields.append('volume')
    if with_amt:
        fields.append('turnover')
    raw_rows = list(qs.values(*fields))
    # 转换数值为基础类型，避免前端图表无法渲染
    data = []
    for r in raw_rows:
        factor = float(r.get('adjust_factor') or 1)
        o = float(r['open'])
        h = float(r['high'])
        l = float(r['low'])
        c = float(r['close'])
        if hfq:
            o, h, l, c = o*factor, h*factor, l*factor, c*factor
        item = {
            'trade_date': r['trade_date'],
            'open': o,
            'high': h,
            'low': l,
            'close': c,
        }
        if with_vol:
            item['volume'] = int(r.get('volume') or 0)
        if with_amt:
            v = r.get('turnover')
            item['turnover'] = float(v) if v is not None else None
        data.append(item)
    if ma and data:
        try:
            ma_list = [int(x) for x in ma.split(',') if x.strip()]
            closes = [d['close'] for d in data]
            for m in ma_list:
                ma_vals = []
                for i in range(len(closes)):
                    if i + 1 < m:
                        ma_vals.append(None)
                    else:
                        window = closes[i + 1 - m:i + 1]
                        ma_vals.append(sum(window) / m)
                key = f'ma{m}'
                for i in range(len(data)):
                    data[i][key] = ma_vals[i]
        except Exception:
            pass
    # 评分线（final_score）改为从因子表取 ML_STOCK_SCORE，按当日对齐
    if data:
        try:
            score_qs = (
                DailyFactorValues.objects
                .filter(
                    stock_code_id=code,
                    factor_code_id='ML_STOCK_SCORE',
                    trade_date__gte=s, trade_date__lte=e
                )
                .values('trade_date', 'norm_score', 'raw_value')
            )
            score_map = {}
            for row in score_qs:
                val = row.get('norm_score')
                if val is None:
                    val = row.get('raw_value')
                try:
                    score_map[row['trade_date']] = float(val) if val is not None else None
                except Exception:
                    score_map[row['trade_date']] = None

            for item in data:
                item['final_score'] = score_map.get(item['trade_date'])
        except Exception:
            for item in data:
                item['final_score'] = item.get('final_score', None)
    if with_m:
        pass
    return ok(data)


@require_http_methods(["POST"])
def daily_fetch(request):
    try:
        import json
        payload = json.loads(request.body.decode('utf-8') or '{}')
    except Exception:
        payload = {}
    codes = payload.get('codes')
    start = payload.get('start')
    end = payload.get('end')
    fill_missing_for_date = payload.get('fill_missing_for_date')
    if not start or not end:
        return err("缺少参数: start 或 end")

    svc = StockService()
    target_codes = None
    if fill_missing_for_date:
        try:
            dt = datetime.strptime(fill_missing_for_date, '%Y-%m-%d').date()
        except Exception:
            return err("参数 fill_missing_for_date 格式应为 YYYY-MM-DD")
        stocks = set(StockInfo.objects.values_list('stock_code', flat=True))
        have_quotes = set(
            DailyQuotes.objects.filter(trade_date=dt).values_list('stock_code_id', flat=True)
        )
        missing = list(stocks - have_quotes)
        target_codes = missing
    elif codes == 'ALL':
        target_codes = list(StockInfo.objects.values_list('stock_code', flat=True))
    elif isinstance(codes, list):
        target_codes = codes

    svc.update_local_a_shares(stock_codes=target_codes, start_date=start, end_date=end)
    return ok({"fetched": True, "size": 0 if target_codes is None else len(target_codes)})


@require_http_methods(["GET"])
def daily_m_value(request):
    start = request.GET.get('start')
    end = request.GET.get('end')
    if not start or not end:
        return err("缺少参数: start 或 end")
    try:
        s = datetime.strptime(start, '%Y-%m-%d').date()
        e = datetime.strptime(end, '%Y-%m-%d').date()
    except Exception:
        return err("参数 start/end 格式应为 YYYY-MM-DD")
    qs = DailyFactorValues.objects.filter(
        stock_code_id='_MARKET_REGIME_INDICATOR_',
        factor_code_id='dynamic_M_VALUE',
        trade_date__gte=s, trade_date__lte=e
    ).order_by('trade_date').values('trade_date', 'raw_value')
    return ok(list(qs))


# ----------------------- 因子管理 -----------------------
@require_http_methods(["GET", "POST", "PUT", "DELETE"])
def factors_params(request):
    if request.method == 'GET':
        items = StrategyParameters.objects.all().values('param_name', 'param_value', 'group_name', 'description')
        return ok(list(items))
    try:
        import json
        payload = json.loads(request.body.decode('utf-8') or '{}')
    except Exception:
        payload = {}
    if request.method == 'POST':
        if not payload.get('param_name'):
            return err('缺少参数: param_name')
        StrategyParameters.objects.update_or_create(
            param_name=payload['param_name'],
            defaults={
                'param_value': Decimal(str(payload.get('param_value', '0'))),
                'group_name': payload.get('group_name') or None,
                'description': payload.get('description') or None,
            }
        )
        return ok({"saved": True})
    if request.method == 'PUT':
        if not payload.get('param_name'):
            return err('缺少参数: param_name')
        StrategyParameters.objects.update_or_create(
            param_name=payload['param_name'],
            defaults={
                'param_value': Decimal(str(payload.get('param_value', '0'))),
                'group_name': payload.get('group_name') or None,
                'description': payload.get('description') or None,
            }
        )
        return ok({"saved": True})
    if request.method == 'DELETE':
        name = request.GET.get('name')
        if not name:
            return err('缺少参数: name')
        StrategyParameters.objects.filter(param_name=name).delete()
        return ok({"deleted": True})


@require_http_methods(["GET", "POST", "PUT", "DELETE"])
def factors_definitions(request):
    if request.method == 'GET':
        items = FactorDefinitions.objects.all().values('factor_code', 'factor_name', 'description', 'direction', 'is_active')
        return ok(list(items))
    try:
        import json
        payload = json.loads(request.body.decode('utf-8') or '{}')
    except Exception:
        payload = {}
    if request.method == 'POST':
        if not payload.get('factor_code'):
            return err('缺少参数: factor_code')
        FactorDefinitions.objects.update_or_create(
            factor_code=payload['factor_code'],
            defaults={
                'factor_name': payload.get('factor_name', ''),
                'description': payload.get('description') or None,
                'direction': payload.get('direction', 'positive'),
                'is_active': bool(payload.get('is_active', True))
            }
        )
        return ok({"saved": True})
    if request.method == 'PUT':
        if not payload.get('factor_code'):
            return err('缺少参数: factor_code')
        FactorDefinitions.objects.update_or_create(
            factor_code=payload['factor_code'],
            defaults={
                'factor_name': payload.get('factor_name', ''),
                'description': payload.get('description') or None,
                'direction': payload.get('direction', 'positive'),
                'is_active': bool(payload.get('is_active', True))
            }
        )
        return ok({"saved": True})
    if request.method == 'DELETE':
        code = request.GET.get('code')
        if not code:
            return err('缺少参数: code')
        FactorDefinitions.objects.filter(factor_code=code).delete()
        return ok({"deleted": True})


# ----------------------- 系统管理 -----------------------
@require_http_methods(["GET", "DELETE"])
def system_schema(request):
    """
    GET  列出可删除的 schema（默认前缀 m_dist_）
      - 可选参数：prefix（默认 m_dist_）
    DELETE 删除指定 schema：严格白名单与名称校验，执行 DROP SCHEMA <name> CASCADE
      - 参数：name
    """
    if request.method == 'GET':
        # 支持逗号分隔的多个前缀，默认 m_dist_ 与 backtest_
        prefix = request.GET.get('prefix', 'm_dist_,backtest_')
        prefixes = [p.strip() for p in prefix.split(',') if p.strip()]
        if not prefixes:
            prefixes = ['m_dist_']
        try:
            with connection.cursor() as cur:
                conditions = " OR ".join(["schema_name LIKE %s"] * len(prefixes))
                sql = f"""
                    SELECT schema_name
                    FROM information_schema.schemata
                    WHERE {conditions}
                    ORDER BY schema_name DESC
                    LIMIT 500
                """
                params = [p + '%' for p in prefixes]
                cur.execute(sql, params)
                rows = [r[0] for r in cur.fetchall()]
            return ok(rows)
        except Exception as e:
            return err(f"查询 schema 列表失败: {e}")

    # DELETE
    name = request.GET.get('name')
    if not name:
        return err('缺少参数: name')

    # 严格校验：仅允许小写字母/数字/下划线，且必须以允许前缀开头
    import re
    if not re.match(r'^[a-z_][a-z0-9_]*$', name or ''):
        return err('非法的 schema 名称')
    allowed_prefixes = ['m_dist_', 'backtest_']
    if not any(name.startswith(p) for p in allowed_prefixes):
        return err('不在允许的 schema 前缀白名单内')

    # 保护系统/常用 schema
    protected = {'public', 'pg_catalog', 'information_schema'}
    if name in protected:
        return err('禁止删除系统 schema')

    try:
        with connection.cursor() as cur:
            # 直接使用经严格校验后的名称，避免 SQL 注入风险
            cur.execute(f'DROP SCHEMA {name} CASCADE')
        return ok({"deleted": True, "name": name})
    except Exception as e:
        return err(f"删除失败: {e}")


@require_http_methods(["GET"])
def system_backtest_results(request):
    """
    回测结果查询（普通回测优先）：
    - 参数：schema（必填，backtest_* 为普通回测），start/end（可选），rf（年化无风险利率，默认0.02）
    - 返回：资金曲线、M值、Sharpe(逐日累计)、以及汇总指标（年化收益、最大回撤、Sharpe）
    """
    schema = request.GET.get('schema')
    if not schema:
        return err('缺少参数: schema')
    rf_annual = request.GET.get('rf')
    try:
        rf_annual = float(rf_annual) if rf_annual is not None else 0.02
    except Exception:
        rf_annual = 0.02

    # 仅处理普通回测
    if not schema.startswith('backtest_'):
        return ok({"mode": "unsupported", "schema": schema})

    # 校验schema 名称
    import re
    if not re.match(r'^[a-z_][a-z0-9_]*$', schema or ''):
        return err('非法的 schema 名称')

    # 先取该schema中可用日期范围
    try:
        with connection.cursor() as cur:
            cur.execute(f'SELECT MIN(trade_date), MAX(trade_date) FROM {schema}.tb_trade_manager_backtest_daily_log')
            row = cur.fetchone()
            if not row or not row[0] or not row[1]:
                return ok({"schema": schema, "mode": "normal", "dates": [], "equity": [], "m_value": [], "csi300": [], "sharpe": [], "summary": {"annualized": 0, "max_drawdown": 0, "sharpe": 0}, "range": None})
            min_date, max_date = row[0], row[1]
    except Exception as e:
        return err(f"读取回测日期范围失败: {e}")

    # 解析 start/end，默认全区间
    start_str = request.GET.get('start')
    end_str = request.GET.get('end')
    try:
        s = datetime.strptime(start_str, '%Y-%m-%d').date() if start_str else min_date
    except Exception:
        s = min_date
    try:
        e = datetime.strptime(end_str, '%Y-%m-%d').date() if end_str else max_date
    except Exception:
        e = max_date
    if s > e:
        s, e = e, s

    # 查询资金与M值
    with connection.cursor() as cur:
        cur.execute(
            f"""
            SELECT trade_date, total_assets, market_m_value
            FROM {schema}.tb_trade_manager_backtest_daily_log
            WHERE trade_date >= %s AND trade_date <= %s
            ORDER BY trade_date
            """,
            [s, e]
        )
        rows = cur.fetchall()

    if not rows:
        return ok({"schema": schema, "mode": "normal", "dates": [], "equity": [], "m_value": [], "csi300": [], "sharpe": [], "summary": {"annualized": 0, "max_drawdown": 0, "sharpe": 0}, "range": {"min": min_date.isoformat(), "max": max_date.isoformat()}})

    # 计算指标
    dates = []
    equity = []
    m_values = []
    for d, ta, mv in rows:
        dates.append(d.isoformat())
        try:
            equity.append(float(ta))
        except Exception:
            equity.append(None)
        try:
            m_values.append(float(mv) if mv is not None else None)
        except Exception:
            m_values.append(None)

    # 获取沪深300收盘价序列（用于前端对齐为同起点对照）
    csi_map = {}
    try:
        csi_qs = (
            IndexQuotesCsi300.objects
            .filter(trade_date__gte=s, trade_date__lte=e)
            .order_by('trade_date')
            .values('trade_date', 'close')
        )
        for r in csi_qs:
            td = r['trade_date']
            try:
                csi_map[td.isoformat()] = float(r['close']) if r['close'] is not None else None
            except Exception:
                csi_map[td.isoformat()] = None
    except Exception:
        csi_map = {}
    csi_series = [csi_map.get(d) for d in dates]

    # 日收益率
    import math
    daily_returns = []
    for i in range(1, len(equity)):
        prev = equity[i-1]
        curr = equity[i]
        if prev and prev != 0 and curr is not None:
            daily_returns.append((curr / prev) - 1.0)
        else:
            daily_returns.append(0.0)

    # 累计Sharpe（从区间起始滚动到当日），rf按252交易日折算
    sharpe_series = []
    rf_daily = (rf_annual or 0.0) / 252.0
    import statistics
    for i in range(len(daily_returns)):
        sub = daily_returns[:i+1]
        if len(sub) < 2 or all(abs(x - rf_daily) < 1e-12 for x in sub):
            sharpe_series.append(0.0)
            continue
        mean_excess = statistics.fmean([x - rf_daily for x in sub])
        std = statistics.pstdev(sub)
        sharpe = (math.sqrt(252.0) * mean_excess / std) if std > 0 else 0.0
        sharpe_series.append(sharpe)
    # 对齐长度（与dates一致，首日无收益设0）
    sharpe_series = [0.0] + sharpe_series

    # 汇总：年化、最大回撤、Sharpe（区间整体）
    n = len(equity)
    if n >= 2 and equity[0] and equity[-1]:
        ann = (equity[-1] / equity[0]) ** (252.0 / (n - 1)) - 1.0
    else:
        ann = 0.0
    # 最大回撤
    peak = -float('inf')
    max_dd = 0.0
    for val in equity:
        if val is None:
            continue
        peak = max(peak, val)
        if peak > 0:
            dd = (val / peak) - 1.0
            max_dd = min(max_dd, dd)
    # 区间Sharpe
    if len(daily_returns) >= 2:
        mean_excess = statistics.fmean([x - rf_daily for x in daily_returns])
        std = statistics.pstdev(daily_returns)
        sharpe_all = (math.sqrt(252.0) * mean_excess / std) if std > 0 else 0.0
    else:
        sharpe_all = 0.0

    return ok({
        "schema": schema,
        "mode": "normal",
        "start": s.isoformat(),
        "end": e.isoformat(),
        "range": {"min": min_date.isoformat(), "max": max_date.isoformat()},
        "dates": dates,
        "equity": equity,
        "m_value": m_values,
        "csi300": csi_series,
        "sharpe": sharpe_series,
        "summary": {"annualized": ann, "max_drawdown": max_dd, "sharpe": sharpe_all},
        "rf": rf_annual
    })

