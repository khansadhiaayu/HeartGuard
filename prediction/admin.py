
from django.contrib import admin
from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):

    list_display = (
        'id',
        'age',
        'sex',
        'chest_pain_type',
        'prediction_result',
        'prediction_date',
    )

    list_filter = (
        'prediction_result',
        'sex',
        'prediction_date',
    )

    search_fields = (
        'sex',
        'chest_pain_type',
    )

    ordering = ('-prediction_date',)
