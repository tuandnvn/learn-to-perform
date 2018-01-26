import json

from django.shortcuts import render
# Create your views here.
from django.http import HttpResponse

def index(request):
    if request.method == 'POST':
        # json_data = json.loads(request.body)

        json_data = json.loads(request.body.decode('utf-8'))

        print (json_data)
    elif request.method == 'GET':
        pass
    return HttpResponse("slide(block1,<-0.56; 0.80; 0.18>)")