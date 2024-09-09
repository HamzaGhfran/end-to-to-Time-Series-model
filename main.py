from steps import ingest, preprocess, train, evaluate


if __name__ == "__main__":
    data = ingest.ingest_fn()
    processed_data = preprocess.preprocess_fn(data)
    train.estimator_fn(processed_data)
    #mse, rmse, mean_error, results = evaluate.evaluate_fn(model, test_data, eval_df_noexog)
    #model, test_data, eval_df_noexog