from pipelines.engineered_metrics_svm.pipeline import SVMPipeline

eng_svm = SVMPipeline('data/image_set_73')
eng_svm.train()
eng_svm.test()
pass