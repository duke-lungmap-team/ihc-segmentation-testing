from .common_utils import utils
import abc


class BasePipeline(object):
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
    def test(self, saved_region_parent_dir=None):
        pass

    def generate_report(self):
        self.report = utils.find_overlapping_regions(
            self.training_data[self.test_img_name],
            self.test_results
        )

    def mean_avg_precision(self):
        iou_mat, pred_mat = utils.generate_iou_pred_matrices(
            self.training_data[self.test_img_name],
            self.test_results
        )

        tp, fn, fp = utils.generate_tp_fn_fp(iou_mat, pred_mat, iou_thresh=0.33, pred_thresh=0.25)

        df, results = utils.generate_dataframe_aggregation_tp_fn_fp(
            self.training_data[self.test_img_name],
            self.test_results,
            iou_mat,
            pred_mat,
            tp,
            fn,
            fp
        )

        # add ground truth label to processed test data frame
        # only valid for true positives and false positives
        # (the missed items have no corresponding test region)
        self.test_data_processed['true_label'] = ''
        for struct_class, struct_dict in results.items():
            for res_dict in struct_dict['fp']:
                overlaps = utils.find_overlapping_regions(
                    self.training_data[self.test_img_name],
                    [self.test_results[res_dict['test_ind']]]
                )['overlaps']

                if len(overlaps) > 0:
                    match = overlaps[list(overlaps.keys())[0]]
                    true_label = match['true_label']
                else:
                    true_label = ''

                self.test_data_processed.set_value(
                    res_dict['test_ind'],
                    'true_label',
                    true_label
                )
                self.test_data_processed.set_value(
                    res_dict['test_ind'],
                    'iou',
                    res_dict['iou']
                )
            for res_dict in struct_dict['tp']:
                if 'gt_ind' in res_dict:
                    continue
                self.test_data_processed.set_value(
                    res_dict['test_ind'],
                    'true_label',
                    struct_class
                )
                self.test_data_processed.set_value(
                    res_dict['test_ind'],
                    'iou',
                    res_dict['iou']
                )

        return df, results

    def plot_results(self):
        if self.report is None:
            raise UserWarning("There are no results to plot, have you run report()?")
            return

        utils.plot_results(self.training_data[self.test_img_name]['hsv_img'], self.report)
