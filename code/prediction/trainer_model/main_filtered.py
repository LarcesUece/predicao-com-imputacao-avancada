from improvement_filtered_links import ImprovementLinks

if __name__ == "__main__":
    improver = ImprovementLinks(
        path_links="../../../datasets/imputed-series2",
        summary_csv="simple_comparison_summary.csv",
        detail_csv="simple_models_detail.csv",
        top_k=5,
        selection_mode="lowest_improvement",  # pega os piores para tentar melhorar
        improvement_threshold=None,  # ex.: 0 para pegar só os que não melhoraram
        cv_splits=5,
    )
    improver.run()
