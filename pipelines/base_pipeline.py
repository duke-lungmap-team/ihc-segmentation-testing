from .common_utils import utils
import abc
import matplotlib.pyplot as plt
import os

# this is just to un-confuse pycharm
try:
    from cv2 import cv2
except ImportError:
    import cv2


class BasePipeline(object):
    def __init__(self, image_set_dir, test_img_index=0):
        # get labeled ground truth regions from given image set
        self.training_data = utils.get_training_data_for_image_set(image_set_dir)
        self.test_img_name = sorted(self.training_data.keys())[test_img_index]
        self.test_marker_set = set(self.test_img_name.split('_')[-4:-1])
        self.test_results = None
        self.report = None
        self.test_data_processed = None

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def test(self, saved_region_parent_dir=None):
        pass

    def generate_report(self):
        iou_mat, pred_mat = utils.generate_iou_pred_matrices(
            self.training_data[self.test_img_name],
            self.test_results
        )

        tp, fn, fp = utils.generate_tp_fn_fp(
            iou_mat,
            pred_mat,
            iou_thresh=0.33,
            pred_thresh=0.50
        )

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

        self.report = results

        return df

    def plot_results(self, save_dir=None):
        if self.report is None:
            raise UserWarning(
                "There are no results to plot, have you run report()?"
            )

        hsv_img = self.training_data[self.test_img_name]['hsv_img'].copy()
        ground_truth = self.training_data[self.test_img_name]['regions']
        test_results = self.test_results
        pipeline_name = type(self).__name__

        # first, map ground truth indices by label
        gt_by_label_map = {}
        for i, gt in enumerate(ground_truth):
            if gt['label'] not in gt_by_label_map:
                gt_by_label_map[gt['label']] = []

            gt_by_label_map[gt['label']].append(gt['points'])

        tp_by_label_map = {}
        fp_by_label_map = {}
        fn_by_label_map = {}
        for k, v in self.report.items():
            if k not in tp_by_label_map:
                tp_by_label_map[k] = []
                fp_by_label_map[k] = []
                fn_by_label_map[k] = []

            for tp in v['tp']:
                if 'test_ind' in tp:
                    tp_by_label_map[k].append(
                        test_results[tp['test_ind']]['contour']
                    )

            for fp in v['fp']:
                if 'test_ind' in fp:
                    fp_by_label_map[k].append(
                        test_results[fp['test_ind']]['contour']
                    )

            for fn in v['fn']:
                if 'gt_ind' in fn:
                    fn_by_label_map[k].append(
                        ground_truth[fn['gt_ind']]['points']
                    )

        # create separate set of images for each class label
        for class_label in sorted(self.report.keys()):
            # ground truth
            if not class_label.startswith('background'):
                new_img = cv2.cvtColor(hsv_img.copy(), cv2.COLOR_HSV2RGB)
                cv2.drawContours(new_img, gt_by_label_map[class_label], -1, (0, 255, 0), 7)
                plt.figure(figsize=(8, 8))
                plt.imshow(new_img)
                title = "%s - %s - 01 - %s" % (pipeline_name, class_label, 'Ground Truth')
                plt.title(title)

                if save_dir is not None:
                    plt.savefig(
                        os.path.join(save_dir, title + '.png'),
                        bbox_inches='tight'
                    )
                else:
                    plt.show()

            # true positive
            new_img = cv2.cvtColor(hsv_img.copy(), cv2.COLOR_HSV2RGB)
            cv2.drawContours(new_img, tp_by_label_map[class_label], -1, (0, 255, 0), 7)
            plt.figure(figsize=(8, 8))
            plt.imshow(new_img)
            title = "%s - %s - 02 - %s" % (pipeline_name, class_label, 'True Positive')
            plt.title(title)
            if save_dir is not None:
                plt.savefig(
                    os.path.join(save_dir, title + '.png'),
                    bbox_inches='tight'
                )
            else:
                plt.show()

            # false negative
            new_img = cv2.cvtColor(hsv_img.copy(), cv2.COLOR_HSV2RGB)
            cv2.drawContours(new_img, fn_by_label_map[class_label], -1, (0, 255, 0), 7)
            plt.figure(figsize=(8, 8))
            plt.imshow(new_img)
            title = "%s - %s - 03 - %s" % (pipeline_name, class_label, 'False Negative')
            plt.title(title)
            if save_dir is not None:
                plt.savefig(
                    os.path.join(save_dir, title + '.png'),
                    bbox_inches='tight'
                )
            else:
                plt.show()

            # false positive
            new_img = cv2.cvtColor(hsv_img.copy(), cv2.COLOR_HSV2RGB)
            cv2.drawContours(new_img, fp_by_label_map[class_label], -1, (0, 255, 0), 7)
            plt.figure(figsize=(8, 8))
            plt.imshow(new_img)
            title = "%s - %s - 04 - %s" % (pipeline_name, class_label, 'False Positive')
            plt.title(title)
            if save_dir is not None:
                plt.savefig(
                    os.path.join(save_dir, title + '.png'),
                    bbox_inches='tight'
                )
            else:
                plt.show()
            plt.close()
