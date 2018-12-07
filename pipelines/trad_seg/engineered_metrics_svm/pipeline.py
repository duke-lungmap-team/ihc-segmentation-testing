import numpy as np
import pandas as pd
import os
from cv_color_features import utils as color_utils
from cv2_extras import utils as cv2x
from pipelines.base_pipeline import BasePipeline
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from pipelines.trad_seg.utils import process_image, generate_color_contours, \
    generate_saturation_contours

# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2


class SVMPipeline(BasePipeline):
    def __init__(self, image_set_dir, test_img_index=0):
        super(SVMPipeline, self).__init__(image_set_dir, test_img_index)
        classifier = SVC(
            kernel='rbf',
            probability=True,
            gamma='scale',
            class_weight='balanced',
        )
        self.pipe = Pipeline(
            [
                ('scaler', MinMaxScaler()),
                ('classification', classifier)
            ]
        )
        self.training_data_processed = None
        self.test_data_processed = None

    def train(self):
        # process the training data to get custom metrics
        training_data_processed = []

        for img_name, img_data in self.training_data.items():
            if img_name == self.test_img_name:
                continue

            non_bg_regions = []

            for region in img_data['regions']:
                training_data_processed.append(
                    color_utils.generate_features(
                        hsv_img_as_numpy=img_data['hsv_img'],
                        polygon_points=region['points'],
                        label=region['label']
                    )
                )

                non_bg_regions.append(region['points'])

            # only generate background regions for images with markers
            # matching the test image
            marker_set = set(img_name.split('_')[-4:-1])
            if self.test_marker_set != marker_set:
                continue

            bg_contours = cv2x.generate_background_contours(
                img_data['hsv_img'],
                non_bg_regions
            )

            for region in bg_contours:
                training_data_processed.append(
                    color_utils.generate_features(
                        hsv_img_as_numpy=img_data['hsv_img'],
                        polygon_points=region,
                        label='background'
                    )
                )

        # train the SVM model with the ground truth
        self.training_data_processed = pd.DataFrame(training_data_processed)

        # drop the dark blue features
        self.training_data_processed.drop(
            list(self.training_data_processed.filter(regex='dark_blue')),
            axis=1,
            inplace=True
        )

        self.pipe.fit(
            self.training_data_processed.drop('label', axis=1),
            self.training_data_processed['label']
        )

    def test(self, saved_region_parent_dir=None):
        img_hsv = self.training_data[self.test_img_name]['hsv_img']
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        img_s = img_hsv[:, :, 1]
        img_shape = (
            img_hsv.shape[0],
            img_hsv.shape[1]
        )

        # blur kernels
        large_blur_kernel = (95, 95)
        med_blur_kernel = (63, 63)
        small_blur_kernel = (31, 31)

        print("Pre-processing test data...")

        b_suppress_img, edge_mask = process_image(
            self.training_data[self.test_img_name]['hsv_img']
        )

        # color candidates (all non-monochrome, non-blue colors)
        print("Generating color candidates...")

        b_suppress_img_hsv = cv2.cvtColor(b_suppress_img, cv2.COLOR_RGB2HSV)
        final_color_contours = generate_color_contours(
            b_suppress_img_hsv,
            edge_mask
        )

        print("%d color candidates found" % len(final_color_contours))

        # final color mask to use for exclusion from further groups
        final_color_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(final_color_mask, final_color_contours, -1, 255, -1)

        # start large structure saturation candidates
        print("Generating large saturation candidates...")

        min_size = 10 * 32 * 32
        max_size = (img_shape[0] * img_shape[1]) * .33
        max_dilate_percentage = 2.0

        final_large_sat_val_contours, edge_mask = generate_saturation_contours(
            img_s,
            large_blur_kernel,
            ~final_color_mask,
            edge_mask,
            min_size,
            max_size,
            erode_iter=5,
            max_dilate_percentage=max_dilate_percentage
        )

        print("%d large sat candidates found" % len(final_large_sat_val_contours))

        # final group mask to use for exclusion from further groups
        final_large_sat_val_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(
            final_large_sat_val_mask,
            final_large_sat_val_contours,
            -1,
            255,
            -1
        )

        # start medium structure saturation candidates
        print("Generating medium saturation candidates...")

        min_size = 2 * 32 * 32
        max_size = (img_shape[0] * img_shape[1]) * .125
        max_dilate_percentage = 1.5

        # make intermediate candidate mask
        tmp_candidate_mask = np.bitwise_or(
            final_color_mask,
            final_large_sat_val_mask
        )

        final_med_sat_val_contours, edge_mask = generate_saturation_contours(
            img_s,
            med_blur_kernel,
            ~tmp_candidate_mask,
            edge_mask,
            min_size,
            max_size,
            erode_iter=8,
            max_dilate_percentage=max_dilate_percentage
        )

        print("%d medium sat candidates found" % len(final_med_sat_val_contours))

        # final color mask to use for exclusion from further groups
        final_med_sat_val_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(final_med_sat_val_mask, final_med_sat_val_contours, -1, 255, -1)

        # start small structure saturation candidates
        print("Generating small saturation candidates...")

        min_size = 15 * 15
        max_size = (img_shape[0] * img_shape[1]) * .05
        max_dilate_percentage = 2.5

        # make intermediate candidate mask
        tmp_candidate_mask = np.bitwise_or(
            tmp_candidate_mask,
            final_med_sat_val_mask
        )

        # TODO: orig small re-dilate was 3 iterations
        small_sat_val_contours, edge_mask = generate_saturation_contours(
            img_s,
            blur_type='bilateral',
            blur_kernel=small_blur_kernel,
            mask=~tmp_candidate_mask,
            edge_mask=edge_mask,
            min_size=min_size,
            max_size=max_size,
            erode_iter=2,
            max_dilate_percentage=max_dilate_percentage
        )

        final_small_sat_val_contours = []

        for c in small_sat_val_contours:
            area = cv2.contourArea(c)
            if area > 27 * 27:
                final_small_sat_val_contours.append(c)

        print("%d small sat candidates found" % len(final_small_sat_val_contours))

        all_contours = final_color_contours + \
            final_large_sat_val_contours + \
            final_med_sat_val_contours + \
            final_small_sat_val_contours

        print("%d total candidates found" % len(all_contours))

        model_classes = list(self.pipe.named_steps['classification'].classes_)
        final_contour_results = []

        test_data_processed = []

        for c_idx, c in enumerate(all_contours):
            if saved_region_parent_dir is not None:
                region_base_name = '.'.join([str(c_idx), 'png'])
                region_file_path = os.path.join(saved_region_parent_dir, region_base_name)

                if not os.path.exists(saved_region_parent_dir):
                    os.makedirs(saved_region_parent_dir)
            else:
                region_file_path = None
                region_base_name = None
            features = color_utils.generate_features(
                img_hsv,
                c,
                region_file_path=region_file_path
            )
            features_df = pd.DataFrame([features])

            # drop the dark blue features
            features_df.drop(
                list(features_df.filter(regex='dark_blue')),
                axis=1,
                inplace=True
            )

            probabilities = self.pipe.predict_proba(features_df.drop('label', axis=1))
            labelled_probs = {a: probabilities[0][i] for i, a in enumerate(model_classes)}
            pred_label = max(labelled_probs, key=lambda key: labelled_probs[key])
            pred_label_prob = labelled_probs[pred_label]

            final_contour_results.append(
                {
                    'test_index': c_idx,
                    'contour': c,
                    'prob': labelled_probs
                }
            )

            features['pred_label'] = pred_label
            features['pred_label_prob'] = pred_label_prob
            features['region_file'] = region_base_name
            features['region_index'] = c_idx

            test_data_processed.append(features)

        # final contour mask for viewing residual areas
        final_contour_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(final_contour_mask, all_contours, -1, 255, -1)

        residual_image = cv2.bitwise_and(
            img_rgb,
            img_rgb,
            mask=~final_contour_mask
        )
        residual_image_path = os.path.join(
            saved_region_parent_dir,
            os.path.splitext(self.test_img_name)[0] + '_residual.tif'
        )
        cv2.imwrite(
            residual_image_path,
            cv2.cvtColor(residual_image, cv2.COLOR_RGB2BGR)
        )
        final_contour_path = os.path.join(
            saved_region_parent_dir,
            os.path.splitext(self.test_img_name)[0] + '_mask.npy'
        )
        np.save(final_contour_path, final_contour_mask)

        self.test_data_processed = pd.DataFrame(test_data_processed)
        self.test_data_processed.set_index('region_index')

        self.test_results = final_contour_results
