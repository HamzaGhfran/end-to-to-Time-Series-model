import pandas as pd
import numpy as np
import mlflow
import mlflow.statsmodels
from mlflow.models import infer_signature
from urllib.parse import urlparse
from mlflow.models import infer_signature
from statsforecast import StatsForecast
from utilsforecast.losses import smape, rmse, mae
from utilsforecast.evaluation import evaluate
import matplotlib.pyplot as plt
from ingest import ingest_fn
from utility.modelwrap import StatsForecastModel
from preprocess import preprocess_fn
from evaluate import evaluate_fn
import yaml
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
        n_windows= 2
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




def estimator_fn(data_path):

    data = pd.read_csv(data_path)
    
    #data.drop([ 'Unnamed: 0','hospital', 'medicine_type'], axis=1, inplace=True)
    #train and test split
    train = data[data['ds']<='2023-12-01']
    test = data[data['ds']>='2024-01-01']

    #crossval_df_exog, eval_df_exog = crossvalidate_and_evaluate(df=train, metric=smape, exog=True, plot=True)
    crossval_df_noexog, eval_df_noexog = crossvalidate_and_evaluate(df=train, metric=smape, exog=False, plot=True)

    train['hospital'] = train['unique_id'].apply(lambda x: x.split('_')[0])
    train['medicine_type'] = train['unique_id'].apply(lambda x: x.split('_')[-1].split('#')[0])

    eval_df_noexog.reset_index(inplace=True)

    print(train)
    
    # mlflow.end_run()
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # experiment_name = "stats_models"
    # experiment = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # with mlflow.start_run(experiment_id=experiment) as run:

    mlflow.set_tag("mlflow.runName", "train")

    print("Training Start.....")
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
    print("Evaluation Start...")
    mse, rmse, mean_error, result= evaluate_fn(sf, test, eval_df_noexog)
    print("Evaluation End...")

    print("rmse: ", rmse)
    print("mse: ", mse)
    print("mean_error: ", mean_error)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mean_error", mean_error)

    artifacts_path = "code"

    #project_dir = "./"

    h = 4
    predictions = sf.predict(h=h)
    signature = infer_signature(h, predictions)# predictions = sf.predict(train)

    conda_env = "/home/hamza/BSCS/TSF/conda.yaml"

    sf.save("Forecasting.pkl")

    mlflow.log_artifact("Forecasting.pkl")    

    mlflow.pyfunc.log_model(
        artifact_path=artifacts_path,
        python_model= StatsForecastModel(sf),
        conda_env=conda_env,
        #artifacts = {"code":project_dir},
        registered_model_name="Stats_Model1",
        signature = signature,
        input_example = h
    )


    
        


    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/saadHospitalPharmacy.csv")
    args = parser.parse_args()
    estimator_fn(args.data_path)
