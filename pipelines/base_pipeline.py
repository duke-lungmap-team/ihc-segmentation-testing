from .common_utils.utils import get_training_data_for_image_set
import abc


class Pipeline(object):
    def __init__(self, image_set_dir, test_img_index=0):
        # get labeled ground truth regions from given image set
        self.training_data = get_training_data_for_image_set(image_set_dir)
        self.test_img = sorted(self.training_data.keys())[test_img_index]
        self.test_results = None

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def test(self):
        pass
