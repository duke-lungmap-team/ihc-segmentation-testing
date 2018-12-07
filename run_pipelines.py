import os
import pickle
import pkgutil
from importlib import import_module
import inspect
from pipelines.base_pipeline import BasePipeline

# dynamically load all sub-classes of our base Pipeline class
pipelines = {}

# by default, all pipelines are ignored...comment out to include them
ignore_pipelines = [
    'BasePipeline',
    'SVMPipeline',
    'XGBPipeline',
    'MrCNNPipeline',
    'UnetPipeline',
    'UnetClassifierPipeline'
]

for (ff, name, is_pkg) in pkgutil.iter_modules(['pipelines']):
    new_module = import_module('pipelines.' + name, package='pipelines')

    for i in dir(new_module):
        if i in ignore_pipelines:
            continue
        attribute = getattr(new_module, i)
        if inspect.isclass(attribute) and issubclass(attribute, BasePipeline):
            pipelines[i] = attribute

# get list of image sets
image_set_dirs = os.listdir('data/mm_e16.5_20x_sox9_sftpc_acta2/light_color_corrected')
image_set_dirs = [x for x in image_set_dirs if not x.startswith('.')]

image_set_dirs = ['mm_e16.5_20x_sox9_sftpc_acta2/light_color_corrected']

# make our 'tmp' directory for caching trained & tested pipeline instances
if not os.path.isdir('tmp'):
    os.mkdir('tmp')

for image_set in image_set_dirs:
    for pipe_class_name, pipe_class in pipelines.items():
        pkl_path = os.path.join(
            'tmp',
            '_'.join([image_set, pipe_class_name + '.pkl'])
        )
        pkl_dir = os.path.dirname(pkl_path)
        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir)

        test_img_path = os.path.join(
            'tmp',
            '_'.join([image_set, pipe_class_name])
        )
        image_set_path = os.path.join('data', image_set)

        try:
            pipe_instance = pickle.load(open(pkl_path, 'rb'))
        except (FileNotFoundError, TypeError):
            pipe_instance = pipe_class(image_set_path, 7)
            pipe_instance.train()
            fh = open(pkl_path, 'wb')
            pickle.dump(pipe_instance, fh)
            fh.close()

        pipe_instance.test(
            saved_region_parent_dir=os.path.join(test_img_path, 'regions')
        )
        df = pipe_instance.generate_report()
        pipe_instance.plot_results(save_dir=test_img_path)
