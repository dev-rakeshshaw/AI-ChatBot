from django.urls import path
from . import views

urlpatterns = [
    # GET renders the page; POST on either path will reach the same view
    path("", views.chat, name="chat_page"),
    path("get-response/", views.chat, name="get_response"),
]
