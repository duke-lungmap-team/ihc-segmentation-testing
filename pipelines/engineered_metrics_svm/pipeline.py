import numpy as np
import pandas as pd
import os
from lung_map_utils import utils
from lung_map_utils_extra import utils_extra
from pipelines.base_pipeline import Pipeline

# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2


class SVMPipeline(Pipeline):
    def __init__(self, image_set_dir, test_img_index=0):
        super(SVMPipeline, self).__init__(image_set_dir, test_img_index)
        self.pipe = utils.pipeline
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
                    utils.generate_features(
                        hsv_img_as_numpy=img_data['hsv_img'],
                        polygon_points=region['points'],
                        label=region['label']
                    )
                )

                non_bg_regions.append(region['points'])

            bg_contours = utils_extra.generate_background_contours(
                img_data['hsv_img'],
                non_bg_regions
            )

            for region in bg_contours:
                training_data_processed.append(
                    utils.generate_features(
                        hsv_img_as_numpy=img_data['hsv_img'],
                        polygon_points=region,
                        label='background'
                    )
                )

        # train the SVM model with the ground truth
        self.training_data_processed = pd.DataFrame(training_data_processed)
        self.pipe.fit(
            self.training_data_processed.drop('label', axis=1),
            self.training_data_processed['label']
        )

    def test(self, saved_region_parent_dir=None):
        print("Pre-processing test data...")

        img_hsv = self.training_data[self.test_img_name]['hsv_img']
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        img_s = img_hsv[:, :, 1]
        img_shape = (img_hsv.shape[0], img_hsv.shape[1])

        # blur kernels
        large_blur_kernel = (95, 95)
        med_blur_kernel = (63, 63)
        small_blur_kernel = (31, 31)

        # Dilation/erosion kernels
        cross_strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        block_strel = np.ones((3, 3))
        ellipse_strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ellipse90_strel = np.rot90(ellipse_strel)
        circle_strel = np.bitwise_or(ellipse_strel, ellipse90_strel)

        b_over_r = img_rgb[:, :, 0] < img_rgb[:, :, 2]
        b_over_g = img_rgb[:, :, 1] < img_rgb[:, :, 2]
        b_over_rg = np.bitwise_and(b_over_r, b_over_g)

        b_replace = np.max([img_rgb[:, :, 0], img_rgb[:, :, 1]], axis=0)

        b_suppress_img = img_rgb.copy()
        b_suppress_img[b_over_rg, 2] = b_replace[b_over_rg]
        b_suppress_img_hsv = cv2.cvtColor(b_suppress_img, cv2.COLOR_RGB2HSV)
        enhanced_v_img = b_suppress_img_hsv[:, :, 2]

        # diff of gaussians
        img_blur_1 = cv2.blur(enhanced_v_img, (15, 15))
        img_blur_2 = cv2.blur(enhanced_v_img, (127, 127))

        tmp_img_1 = img_blur_1.astype(np.int16)
        tmp_img_2 = img_blur_2.astype(np.int16)

        edge_mask = tmp_img_2 - tmp_img_1
        edge_mask[edge_mask > 0] = 0
        edge_mask[edge_mask < 0] = 255

        edge_mask = edge_mask.astype(np.uint8)

        contours = utils_extra.filter_contours_by_size(edge_mask, min_size=15 * 15)

        edge_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(edge_mask, contours, -1, 255, -1)

        edge_mask = cv2.dilate(edge_mask, cross_strel, iterations=1)

        # color candidates (all non-monochrome, non-blue colors)
        print("Generating color candidates...")

        tmp_color_img = cv2.cvtColor(b_suppress_img_hsv, cv2.COLOR_HSV2RGB)
        tmp_color_img = cv2.blur(tmp_color_img, (9, 9))
        tmp_color_img = cv2.cvtColor(tmp_color_img, cv2.COLOR_RGB2HSV)

        color_mask = utils.create_mask(
            tmp_color_img,
            [
                'green', 'cyan', 'red', 'violet', 'yellow'
            ]
        )

        # Next, clean up any "noise", say, ~ 11 x 11
        contours = utils_extra.filter_contours_by_size(color_mask, min_size=5 * 5)

        color_mask2 = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(color_mask2, contours, -1, 255, -1)

        # do a couple dilations to smooth some outside edges & connect any
        # adjacent cells each other
        color_mask3 = cv2.dilate(color_mask2, cross_strel, iterations=2)
        color_mask3 = cv2.erode(color_mask3, cross_strel, iterations=3)

        # filter out any remaining contours that are the size of roughly 2 cells
        contours = utils_extra.filter_contours_by_size(color_mask3, min_size=2 * 40 * 40)

        color_mask3 = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(color_mask3, contours, -1, 255, -1)

        good_color_contours = []

        for i, c in enumerate(contours):
            filled_c_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(filled_c_mask, [c], -1, 255, cv2.FILLED)

            new_mask, signal, orig = utils_extra.find_border_by_mask(
                edge_mask,
                filled_c_mask,
                max_dilate_percentage=0.3,
                dilate_iterations=1
            )

            if not orig and signal > 0.7:
                _, contours, _ = cv2.findContours(
                    new_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                good_color_contours.append(contours[0])
            elif orig and signal > 0.7:
                good_color_contours.append(c)
            else:
                pass  # don't include contour

        # The previous contours represent the final count of contours for the color group,
        # but not their final shape/size. The final step is to use approxPolyDP, followed
        # by a few more dilations, but we won't re-draw and extract the contours, as that
        # would possibly connect some of them together. However, we will draw them for
        # the mask to exclude these regions from the next 2 groups of candidates
        final_color_contours = []

        for c in good_color_contours:
            peri = cv2.arcLength(c, True)
            smooth_c = cv2.approxPolyDP(c, 0.007 * peri, True)

            single_cnt_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(single_cnt_mask, [smooth_c], 0, 255, -1)

            # erode & dilate
            single_cnt_mask = cv2.erode(single_cnt_mask, block_strel, iterations=1)
            single_cnt_mask = cv2.dilate(single_cnt_mask, block_strel, iterations=8)

            _, contours, _ = cv2.findContours(
                single_cnt_mask,
                cv2.RETR_EXTERNAL,

                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) > 0:
                final_color_contours.append(contours[0])

        # final color mask to use for exclusion from further groups
        final_color_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(final_color_mask, final_color_contours, -1, 255, -1)

        # start large structure saturation candidates
        print("Generating large saturation candidates...")

        img_s_large_blur = cv2.GaussianBlur(img_s, large_blur_kernel, 0, 0)

        med = np.median(img_s_large_blur[img_s_large_blur > 0])

        img_s_large_blur = cv2.bitwise_and(
            img_s_large_blur,
            img_s_large_blur,
            mask=~final_color_mask
        )
        mode_s_large = cv2.inRange(img_s_large_blur, med, 255)
        mode_s_large = cv2.erode(mode_s_large, block_strel, iterations=5)

        good_contours_large = utils_extra.filter_contours_by_size(
            mode_s_large,
            min_size=10 * 32 * 32,
            max_size=(img_shape[0] * img_shape[1]) * .33
        )

        # update the signal mask by removing the previous color candidates
        edge_mask = cv2.bitwise_and(edge_mask, edge_mask, mask=~final_color_mask)
        contours = utils_extra.filter_contours_by_size(edge_mask, min_size=21 * 21)
        edge_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(edge_mask, contours, -1, 255, -1)
        edge_mask = cv2.dilate(edge_mask, (3, 3), iterations=2)

        good_large_sat_val_contours = []

        for i, c in enumerate(good_contours_large):
            filled_c_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(filled_c_mask, [c], -1, 255, cv2.FILLED)

            new_mask, signal, orig = utils_extra.find_border_by_mask(
                edge_mask,
                filled_c_mask,
                max_dilate_percentage=2.0,
                dilate_iterations=1
            )

            if not orig and signal > 0.7:
                _, contours, _ = cv2.findContours(
                    new_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                good_large_sat_val_contours.append(contours[0])
            elif signal > 0.7:
                good_large_sat_val_contours.append(c)
            else:
                pass  # ignore contour

        # The previous contours represent the final contour count for the large sat-val group,
        # but not their final shape/size. The final step is to use approxPolyDP, followed
        # by a few more dilations, but we won't re-draw and extract the contours, as that
        # would possibly connect some of them together. However, we will draw them for
        # the mask to exclude these regions from the next group of candidates
        final_large_sat_val_contours = []

        for c in good_large_sat_val_contours:
            peri = cv2.arcLength(c, True)
            smooth_c = cv2.approxPolyDP(c, 0.0035 * peri, True)

            single_cnt_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(single_cnt_mask, [smooth_c], 0, 255, -1)

            # erode & dilate
            single_cnt_mask = cv2.dilate(single_cnt_mask, circle_strel, iterations=10)

            _, tmp_contours, _ = cv2.findContours(
                single_cnt_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            final_large_sat_val_contours.append(tmp_contours[0])

        # final group mask to use for exclusion from further groups
        final_large_sat_val_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(final_large_sat_val_mask, final_large_sat_val_contours, -1, 255, -1)

        # start medium structure saturation candidates
        print("Generating medium saturation candidates...")

        img_s_medium_blur = cv2.GaussianBlur(img_s, med_blur_kernel, 0, 0)
        med = np.median(img_s_medium_blur[img_s_medium_blur > 0])

        # make intermediate candidate mask
        tmp_candidate_mask = np.bitwise_or(final_color_mask, final_large_sat_val_mask)

        img_s_medium_blur = cv2.bitwise_and(
            img_s_medium_blur,
            img_s_medium_blur,
            mask=~tmp_candidate_mask
        )

        mode_s_med = cv2.inRange(img_s_medium_blur, np.ceil(med), 255)

        mode_s_med = cv2.erode(mode_s_med, block_strel, iterations=8)

        good_contours_med = utils_extra.filter_contours_by_size(
            mode_s_med,
            min_size=2 * 32 * 32,
            max_size=(img_shape[0] * img_shape[1]) * .125
        )

        # update signal mask with last group of candidates
        edge_mask = cv2.bitwise_and(edge_mask, edge_mask, mask=~final_large_sat_val_mask)
        contours = utils_extra.filter_contours_by_size(edge_mask, min_size=21 * 21)

        edge_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(edge_mask, contours, -1, 255, -1)
        edge_mask = cv2.dilate(edge_mask, (3, 3), iterations=2)

        final_med_sat_val_contours = []

        for i, c in enumerate(good_contours_med):
            filled_c_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(filled_c_mask, [c], -1, 255, cv2.FILLED)

            new_mask, signal, orig = utils_extra.find_border_by_mask(
                edge_mask,
                filled_c_mask,
                max_dilate_percentage=1.5,
                dilate_iterations=1
            )

            if not orig and signal > 0.7:
                _, contours, _ = cv2.findContours(
                    new_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                final_med_sat_val_contours.append(contours[0])
            elif signal > 0.7:
                final_med_sat_val_contours.append(c)
            else:
                pass  # ignore contour

        med_sat_val_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(med_sat_val_mask, final_med_sat_val_contours, -1, 255, cv2.FILLED)

        # The previous contours represent the final count of contours for the med sat-val group,
        # but not their final shape/size. The final step is to use approxPolyDP, followed
        # by a few more dilations, but we won't re-draw and extract the contours, as that
        # would possibly connect some of them together. However, we will draw them for
        # the mask to exclude these regions from the next group of candidates
        _, contours, _ = cv2.findContours(
            med_sat_val_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        final_med_sat_val_contours = []

        for c in contours:
            peri = cv2.arcLength(c, True)
            smooth_c = cv2.approxPolyDP(c, 0.0035 * peri, True)

            single_cnt_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(single_cnt_mask, [smooth_c], 0, 255, -1)

            # erode & dilate
            single_cnt_mask = cv2.dilate(single_cnt_mask, ellipse90_strel, iterations=10)

            _, tmp_contours, _ = cv2.findContours(
                single_cnt_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            final_med_sat_val_contours.append(tmp_contours[0])

        # final color mask to use for exclusion from further groups
        final_med_sat_val_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(final_med_sat_val_mask, final_med_sat_val_contours, -1, 255, -1)

        # start small structure saturation candidates
        print("Generating small saturation candidates...")

        img_s_small_blur = cv2.bilateralFilter(img_s, small_blur_kernel[0], 31, 31)
        med = np.median(img_s_small_blur[img_s_small_blur > 0])

        # make intermediate candidate mask
        tmp_candidate_mask = np.bitwise_or(tmp_candidate_mask, final_med_sat_val_mask)

        img_s_small_blur = cv2.bitwise_and(
            img_s_small_blur,
            img_s_small_blur,
            mask=~tmp_candidate_mask
        )

        mode_s_small = cv2.inRange(img_s_small_blur, np.ceil(med), 255)

        mode_s_small = cv2.erode(mode_s_small, block_strel, iterations=2)

        good_contours_small = utils_extra.filter_contours_by_size(
            mode_s_small,
            min_size=15 * 15,
            max_size=(img_shape[0] * img_shape[1]) * .05
        )

        # update signal mask with last group of candidates
        edge_mask = cv2.bitwise_and(edge_mask, edge_mask, mask=~final_med_sat_val_mask)
        contours = utils_extra.filter_contours_by_size(edge_mask, min_size=31 * 31)
        edge_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(edge_mask, contours, -1, 255, -1)
        edge_mask = cv2.dilate(edge_mask, (3, 3), iterations=3)

        good_small_sat_val_contours = []

        for i, c in enumerate(good_contours_small):
            filled_c_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(filled_c_mask, [c], -1, 255, cv2.FILLED)

            new_mask, signal, orig = utils_extra.find_border_by_mask(
                edge_mask,
                filled_c_mask,
                max_dilate_percentage=2.5,
                dilate_iterations=1
            )

            if not orig and signal > 0.7:
                _, contours, _ = cv2.findContours(
                    new_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                good_small_sat_val_contours.append(contours[0])
            elif signal > 0.7:
                good_small_sat_val_contours.append(c)
            else:
                pass  # ignore contour

        tmp_small_sat_val_contours = good_small_sat_val_contours.copy()
        good_small_sat_val_contours = []

        for c in tmp_small_sat_val_contours:
            area = cv2.contourArea(c)
            if area > 27 * 27:
                good_small_sat_val_contours.append(c)

        # The previous contours represent the final countour count for the small sat-val group,
        # but not their final shape/size. The final step is to use approxPolyDP, followed
        # by a few more dilations, but we won't re-draw and extract the contours, as that
        # would possibly connect some of them together. However, we will draw them for
        # the mask to exclude these regions from the next group of candidates
        final_small_sat_val_contours = []

        for c in good_small_sat_val_contours:
            peri = cv2.arcLength(c, True)
            smooth_c = cv2.approxPolyDP(c, 0.007 * peri, True)

            single_cnt_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(single_cnt_mask, [smooth_c], 0, 255, -1)

            # erode & dilate
            single_cnt_mask = cv2.dilate(single_cnt_mask, block_strel, iterations=10)

            _, tmp_contours, _ = cv2.findContours(
                single_cnt_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            final_small_sat_val_contours.append(tmp_contours[0])

        all_contours = final_color_contours + \
            final_large_sat_val_contours + \
            final_med_sat_val_contours + \
            final_small_sat_val_contours

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
            features = utils.generate_features(img_hsv, c, region_file_path=region_file_path)
            features_df = pd.DataFrame([features])
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

        self.test_data_processed = pd.DataFrame(test_data_processed)
        self.test_data_processed.set_index('region_index')

        self.test_results = final_contour_results
