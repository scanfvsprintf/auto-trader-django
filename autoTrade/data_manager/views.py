from django.http.response import JsonResponse
from django.shortcuts import render
from common.models import StockInfo
from data_manager.service.stock_service import StockService
from data_manager.service.corporate_action_service import CorporateActionService
from data_manager.service.email_service import EmailNotificationService
from django.views.decorators.http import require_http_methods
import json
from datetime import date,datetime
# Create your views here.
def test_get(request):
    result={}
    if request.method=='GET':
        result=  {'method':'get'}
    if request.method=='POST':
        result= {'methods':'post'}
    # service=StockService()
    # service.clear_all_data()
    # service.update_local_a_shares(start_date="2025-01-01",end_date="2025-08-04")
    # service.update_local_a_shares(start_date="2024-01-01",end_date="2024-12-31")
    # service.update_local_a_shares(start_date="2023-01-01",end_date="2023-12-31")
    # service.update_local_a_shares(start_date="2022-01-01",end_date="2022-12-31")
    # service.update_local_a_shares(start_date="2021-01-01",end_date="2021-12-31")
    return JsonResponse(result)

@require_http_methods(["POST"])
def update_local_a_shares(request):
    body= json.loads(request.body)
    service=StockService()
    service.update_local_a_shares(stock_codes=body['stockCodes'],start_date=body['startDate'],end_date=body['endDate'])
    return JsonResponse({"result":"success"})

@require_http_methods(["POST"])
def sync_corporate_actions(request):
    body= json.loads(request.body)
    service=CorporateActionService()
    service.sync_corporate_actions(start_date=body['startDate'],end_date=body['endDate'])
    return JsonResponse({"result":"success"})

@require_http_methods(["POST"])
def email_send(request):
    body= json.loads(request.body)
    service=EmailNotificationService(t_day=datetime.strptime(body['date'], "%Y-%m-%d").date())
    service.runEmailSend()
    return JsonResponse({"result":"success"})