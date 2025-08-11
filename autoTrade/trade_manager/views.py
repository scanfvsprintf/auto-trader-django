from datetime import date,datetime
from django.shortcuts import render
from django.http.response import JsonResponse
from trade_manager.service.before_fix_service import BeforeFixService
from trade_manager.service.decision_order_service import DecisionOrderService
from trade_manager.service.simulate_trade import SimulateTradeService
import json
from django.views.decorators.http import require_http_methods
# Create your views here.
@require_http_methods(["POST"])
def before_fix_run(request):
    if request.method=='POST':
        body= json.loads(request.body)
        selection_date=datetime.strptime(body['date'], "%Y-%m-%d").date()
        service=BeforeFixService(selection_date)
        service.run()
        return JsonResponse({
            'result':'成功'
        })
@require_http_methods(["GET"])
def initialize_strategy_parameters(request):
    if request.method=='GET':
        DecisionOrderService.initialize_strategy_parameters()
        return JsonResponse({
            'result':'成功'
        })

@require_http_methods(["POST"])
def simulate_trade(request):
    if request.method=='POST':
        body= json.loads(request.body)
        start_date=body['startDate']
        end_date=body['endDate']
        service=SimulateTradeService()
        result=service.run_backtest(start_date=start_date,end_date=end_date)
        return JsonResponse(result)