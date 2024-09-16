import mlflow.pyfunc
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

class StatsForecastModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        # Implement your prediction logic here
        return self.model.predict(model_input)
