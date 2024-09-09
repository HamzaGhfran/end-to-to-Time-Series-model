import pandas as pd
import pickle

# insert model path
# model_path = 'path/to/your/model.pkl'

# with open(model_path, 'rb') as file:
#     model = pickle.load(file)



def predict_fn(model, h):

    forecast = model.predict(h=h)
    forecast.reset_index(inplace=True)
    forecast['ds'] = pd.to_datetime(forecast['ds'])+pd.Timedelta(days=1)
    return forecast







    