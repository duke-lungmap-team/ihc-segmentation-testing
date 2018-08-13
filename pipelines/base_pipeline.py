from .common_utils import utils
import abc


class Pipeline(object):
    def __init__(self, image_set_dir, test_img_index=0):
        # get labeled ground truth regions from given image set
        self.training_data = utils.get_training_data_for_image_set(image_set_dir)
        self.test_img_name = sorted(self.training_data.keys())[test_img_index]
        self.test_results = None
        self.report = None

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def test(self):
        pass

    def report(self):
        self.report = utils.find_overlapping_regions(
            self.training_data[self.test_img_name],
            self.test_results
        )

    def plot_results(self):
        if self.report is None:
            raise UserWarning("There are no results to plot, have you run report()?")
            return

        utils.plot_results(self.training_data[self.test_img_name]['hsv_img'], self.report)
