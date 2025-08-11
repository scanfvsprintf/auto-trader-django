from datetime import date,datetime
from django.shortcuts import render
from django.http.response import JsonResponse
from selection_manager.service.selection_service import SelectionService
import json
from django.views.decorators.http import require_http_methods
# Create your views here.
@require_http_methods(["GET"])
def init_strategy(request):
    SelectionService.initialize_strategy()
    result={}
    return JsonResponse(result)
@require_http_methods(["POST"])
def run_selection(request):
    if request.method=='POST':
        body= json.loads(request.body)
        selection_date=datetime.strptime(body['date'], "%Y-%m-%d").date()
        service=SelectionService(selection_date,mode=body['mode'])
        service.run_selection()
        return JsonResponse({
            'type':selection_date,
            'data':json.loads(request.body)
        })
    
