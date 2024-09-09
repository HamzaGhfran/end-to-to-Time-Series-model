import pandas as pd
import numpy as np
import mlflow
import mlflow.statsmodels
from urllib.parse import urlparse
from mlflow.models import infer_signature
from steps.evaluate import evaluate_fn
from statsforecast import StatsForecast
from utilsforecast.losses import smape, rmse, mae
from utilsforecast.evaluation import evaluate
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


import pandas as pd 
# import model for prediction on dataframe
# implement your train and testing function

def evaluate_crossvalidation(df, metric):
    try:
        models = df.drop(
            columns=['unique_id', 'ds', 'cutoff', 'y']
        ).columns.tolist()
        evals = []

        for cutoff in df['cutoff'].unique():
            eval_ = evaluate(
                df[df['cutoff']==cutoff], metrics=[metric], models=models
            )
            evals.append(eval_)
        evals = pd.concat(evals)
        evals = evals.groupby('unique_id').mean(numeric_only=True)
        evals["best_model"] = evals.idxmin(axis=1)
        return evals
    except Exception as e:
        raise e

def get_best_model_value(row):
    best_model = row['best_model']
    return row[best_model]

def crossvalidate_and_evaluate(df, metric, exog=True, plot=False):

    df = df if exog else df[['ds', 'unique_id', 'y']]
    print(
        f'Crossvalidating with exog: {exog}'
    )
    sf = StatsForecast(
        models=models,
        freq='M', 
        fallback_model = SeasonalNaive(season_length=12, alias='Fallback-SNaive'),
        n_jobs=-1,
    )

    crossvaldation_df = sf.cross_validation(
        df=df,
        h=1,
        step_size=1,
        n_windows=1
    )
    crossvaldation_df.reset_index(inplace=True)

    eval_df = evaluate_crossvalidation(crossvaldation_df, metric)
    eval_df['best_metric'] = eval_df.apply(get_best_model_value, axis=1)
    print(f"Mean metric: {eval_df['best_metric'].mean()}")

    return crossvaldation_df, eval_df
#####################################################################
#####################################################################

from statsforecast.models import ( # type: ignore
MSTL,
AutoARIMA,
AutoCES,
AutoETS,
AutoTheta,
HoltWinters,
RandomWalkWithDrift,
SeasonalNaive,
SeasonalWindowAverage,
DynamicOptimizedTheta as DOT
)

models = [
    # AutoARIMA(season_length=12, nmodels=90, allowdrift=True),
    # AutoETS(season_length=12),
    HoltWinters(season_length=12),
    # MSTL(season_length=12, trend_forecaster=AutoARIMA(), alias='MSTL-ARIMA'),
    # AutoTheta(season_length=12),
    # DOT(season_length=12),
    # SeasonalWindowAverage(window_size=2, season_length=12, alias='SeasWA212'),
    # SeasonalWindowAverage(window_size=1, season_length=12, alias='SeasWA112'),
    # SeasonalWindowAverage(window_size=4, season_length=12, alias='SeasWA412'),
    # RandomWalkWithDrift()
]




def estimator_fn(data):

    
    # data.drop([ 'Unnamed: 0','hospital', 'medicine_type'], axis=1, inplace=True)
    # train and test split
    train = data[data['ds']<='2023-12-01']
    test = data[data['ds']>='2024-01-01']

    #crossval_df_exog, eval_df_exog = crossvalidate_and_evaluate(df=train, metric=smape, exog=True, plot=True)
    crossval_df_noexog, eval_df_noexog = crossvalidate_and_evaluate(df=train, metric=smape, exog=False, plot=True)

    train['hospital'] = train['unique_id'].apply(lambda x: x.split('_')[0])
    train['medicine_type'] = train['unique_id'].apply(lambda x: x.split('_')[-1].split('#')[0])

    eval_df_noexog.reset_index(inplace=True)

    mlflow.statsmodels.autolog()
    with mlflow.start_run():
    # Model Training
        sf = StatsForecast(
        models=models,
        freq='M', 
        fallback_model = SeasonalNaive(season_length=12, alias='Fallback-SNaive'),
        n_jobs=-1,
        )
        sf.fit(
            df = train[['ds', 'unique_id', 'y']],
            prediction_intervals=None
        )

        mse, rmse, mean_error, result = evaluate_fn(sf, test, eval_df_noexog)

        print("rmse: ", rmse)
        print("mse: ", mse)
        print("mean_error: ", mean_error)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mean_error", mean_error)

        # artifact_path = "./mlruns/0"

        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # # Model registry does not work with file store
        # if tracking_url_type_store != "file":

        #     mlflow.statsmodels.log_model(
        #         sf, artifact_path, registered_model_name="TimeSeriesModel"
        #     )
        # else:
        #     mlflow.statsmodels.log_model(sf, artifact_path)

