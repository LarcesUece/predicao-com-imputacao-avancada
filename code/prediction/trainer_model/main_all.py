from prediction_all_links import PredictionAllLinks

predictor = PredictionAllLinks(
    data_dir="../../../datasets/imputed-series2",
    lags=6,
    cv_splits=5,
    min_split_size=30,
    baseline_keep="bfill,ffill,knn,mean,median",
    require_all_baselines=False,
    device="auto",
    window=24,
    same_window=False,
    epochs=25,
    patience=3,
    log_level="INFO",
    log_dir="logs",
    plots=True,
)
if __name__ == "__main__":
    """
      Experiments on all files 
      ----------------------------------
      First, the experiments run in the directory 'imputed_series_selective_unfair', this 
      directory contains 558 links; for each link, there is one stacking file named '<link>_stacking.csv' and five baseline files named '<link>_baseline_<method>.csv' (where <method> is one of bfill, ffill, knn, mean, median).
      The results are saved in 'simple_models_detail.csv' and 'simple_comparison_summary.csv'.
      Afterward, the five best links are selected based on the improvement percentage of stacking over baseline in file 
    """
    predictor = PredictionAllLinks(
        data_dir="../../../datasets/multivariada-imputed-final",
        lags=6,
        cv_splits=5,
        min_split_size=30,
        baseline_keep="bfill,ffill,knn,mean,median",
        require_all_baselines=False,
        device="auto",  # Automatically select CPU when CUDA is unavailable
        window=12,
        same_window=False,
        epochs=25,
        patience=3,
        log_level="INFO",
        log_dir="logs",
        plots=True,
        max_links=200,
    )
    predictor.run()
