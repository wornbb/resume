from django.conf.urls import url
from django.urls import include, path
from django.contrib import admin

from .views import (
    AEAPIView,
)

urlpatterns = [
    path('ae/', AEAPIView.as_view(), name='ae-demo'),
]
