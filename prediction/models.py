
from django.db import models


class PredictionHistory(models.Model):
    age = models.IntegerField()
    sex = models.CharField(max_length=10)
    chest_pain_type = models.CharField(max_length=20)

    resting_bp = models.FloatField()
    cholesterol = models.FloatField()
    fasting_bs = models.IntegerField()

    resting_ecg = models.CharField(max_length=20)
    max_heart_rate = models.IntegerField()
    exercise_angina = models.CharField(max_length=10)

    oldpeak = models.FloatField()
    st_slope = models.CharField(max_length=20)

    prediction_result = models.IntegerField()
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.age} tahun - {self.prediction_result}"
