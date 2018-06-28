import os
import pickle
from pipelines.engineered_metrics_svm.pipeline import SVMPipeline

if not os.path.isdir('tmp'):
    os.mkdir('tmp')

try:
    eng_svm = pickle.load(open('tmp/73_SVMPipeline.pkl', 'rb'))
except (FileNotFoundError, TypeError):
    eng_svm = SVMPipeline('data/image_set_73')
    eng_svm.train()
    eng_svm.test()
    fh = open('tmp/73_SVMPipeline.pkl', 'wb')
    pickle.dump(eng_svm, fh)
    fh.close()

report = eng_svm.report()
