import os
import pickle
import pkgutil
from importlib import import_module
import inspect
from pipelines.base_pipeline import Pipeline

# dynamically load all sub-classes of our base Pipeline class
pipelines = {}

for (ff, name, is_pkg) in pkgutil.iter_modules(['pipelines']):
    if name in ['base_pipeline', 'common_utils']:
        continue

    new_module = import_module('pipelines.' + name, package='pipelines')

    for i in dir(new_module):
        attribute = getattr(new_module, i)
        if inspect.isclass(attribute) and issubclass(attribute, Pipeline):
            pipelines[i] = attribute

# get list of image sets
image_set_dirs = os.listdir('data')

# make our 'tmp' directory for caching trained & tested pipeline instances
if not os.path.isdir('tmp'):
    os.mkdir('tmp')

for image_set in image_set_dirs:
    for pipe_class_name, pipe_class in pipelines.items():
        pkl_path = os.path.join('tmp', '_'.join([image_set, pipe_class_name + '.pkl']))
        image_set_path = os.path.join('data', image_set)

        try:
            pipe_instance = pickle.load(open(pkl_path, 'rb'))
        except (FileNotFoundError, TypeError):
            pipe_instance = pipe_class(image_set_path)
            pipe_instance.train()
            pipe_instance.test()
            fh = open(pkl_path, 'wb')
            pickle.dump(pipe_instance, fh)
            fh.close()

        report = pipe_instance.report()
