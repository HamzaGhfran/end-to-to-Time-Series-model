# Implementation for evaluation function for test data
from predict import predict_fn
import numpy as np
import pandas as pd

def evaluate_fn(model, test, eval_df_noexog):

    forecast = predict_fn(model, 4)
    merged_df = pd.merge(forecast, eval_df_noexog, on='unique_id', suffixes=('_forecast', '_eval'))
    merged_df['model'] = merged_df['best_model'] + '_forecast'
    merged_df['prediction'] = merged_df.apply(lambda row: row[row['model']], axis=1)   

    merged_df = merged_df[['ds','unique_id', 'prediction']]

    test=test[['ds', 'unique_id', 'y']]

    # Change ds to datetime format
    merged_df['ds'] = pd.to_datetime(merged_df['ds'])
    test['ds'] = pd.to_datetime(test['ds']) 
    results = pd.merge(merged_df, test, on=['ds', 'unique_id'])
    
    # Evaluation metric
    mse = np.mean((results['y'] - results['prediction']) ** 2)
    rmse = np.sqrt(mse)
    #print(f'RMSE: {rmse}')

    mean_error = np.mean(results['y'] - results['prediction'])
    #print(mean_error)
    results['adjusted_prediction'] = results['prediction'] - mean_error

    print("Evaluation Done.....")

    return mse, rmse, mean_error, results['adjusted_prediction']