import numpy as np
import pandas as pd
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

    def train(self):
        # process the training data to get custom metrics
        training_data_processed = []

        for img_name, img_data in self.training_data.items():
            if img_name == self.test_img:
                continue
            for region in img_data['regions']:
                training_data_processed.append(
                    utils.generate_features(
                        hsv_img_as_numpy=img_data['hsv_img'],
                        polygon_points=region['points'],
                        label=region['label']
                    )
                )

        # train the SVM model with the ground truth
        training_data_processed = pd.DataFrame(training_data_processed)
        self.pipe.fit(
            training_data_processed.drop('label', axis=1),
            training_data_processed['label']
        )

    def test(self):
        color_blur_kernel = (27, 27)
        large_blur_kernel = (91, 91)

        img = self.training_data[self.test_img]['hsv_img']

        # color threshold candidates
        img_blur_c = cv2.GaussianBlur(
            cv2.cvtColor(img, cv2.COLOR_HSV2RGB),
            color_blur_kernel,
            0
        )
        img_blur_c_hsv = cv2.cvtColor(img_blur_c, cv2.COLOR_RGB2HSV)

        # create masks from feature colors
        mask = utils.create_mask(
            img_blur_c_hsv,
            [
                'cyan',
                'gray',
                'green',
                'red',
                'violet',
                'white',
                'yellow'
            ]
        )

        # define kernel used for erosion & dilation
        kernel = np.ones((3, 3), np.uint8)

        # dilate masks
        mask = cv2.dilate(mask, kernel, iterations=3)

        # fill holes in mask using contours
        mask = utils_extra.fill_holes(mask)

        good_contours_color = utils_extra.filter_contours_by_size(mask, min_size=2048)
        good_contours_color_final = []

        for good_c in good_contours_color:
            hull_c = cv2.convexHull(good_c)
            good_contours_color_final.append(hull_c)

        # start large structure saturation candidates
        img_s = img[:, :, 1]
        img_blur_s_large = cv2.GaussianBlur(img_s, large_blur_kernel, 0)

        lower_bound, upper_bound = utils_extra.determine_hist_mode(img_blur_s_large)

        mode_s_large = cv2.inRange(img_blur_s_large, lower_bound, upper_bound)

        cross_strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        good_contours_large = utils_extra.filter_contours_by_size(
            mode_s_large,
            min_size=200 * 200
        )
        new_good_contours_large = []

        for good_c in good_contours_large:

            single_cnt_mask = np.zeros(mode_s_large.shape, dtype=np.uint8)
            cv2.drawContours(single_cnt_mask, [good_c], 0, 255, -1)

            # dilate single_cnt_mask up to 91 pixels (~47 times)
            single_cnt_mask = cv2.dilate(single_cnt_mask, cross_strel, iterations=47)

            ret, thresh = cv2.threshold(single_cnt_mask, 1, 255, cv2.THRESH_BINARY)
            new_mask, contours, hierarchy = cv2.findContours(
                thresh,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE
            )

            new_good_contours_large.append(contours[0])

        # start small structure saturation candidates
        img_blur_s_small = cv2.GaussianBlur(img_s, (31, 31), 0)
        img_blur_s_small = cv2.bilateralFilter(img_blur_s_small, 31, 31, 255)  # (41, 41), 0)

        mode_s_small1 = cv2.inRange(img_blur_s_small, 240, 256)
        mode_s_small2 = cv2.inRange(img_blur_s_small, 232, 248)
        mode_s_small3 = cv2.inRange(img_blur_s_small, 224, 240)

        min_size = 32 * 32 * 2
        max_size = 200 * 200

        good_contours_small1 = utils_extra.filter_contours_by_size(
            mode_s_small1,
            min_size=min_size,
            max_size=max_size
        )
        good_contours_small2 = utils_extra.filter_contours_by_size(
            mode_s_small2,
            min_size=min_size,
            max_size=max_size
        )
        good_contours_small3 = utils_extra.filter_contours_by_size(
            mode_s_small3,
            min_size=min_size,
            max_size=max_size
        )

        small_contour_sets = [good_contours_small1, good_contours_small2, good_contours_small3]

        new_good_contours_small = []

        for c_set in small_contour_sets:

            for good_c in c_set:
                hull_c = cv2.convexHull(good_c)

                single_cnt_mask = np.zeros(img_s.shape, dtype=np.uint8)
                cv2.drawContours(single_cnt_mask, [hull_c], 0, 255, -1)

                # dilate single_cnt_mask up to 47 pixels (~23 times)
                single_cnt_mask = cv2.dilate(single_cnt_mask, cross_strel, iterations=23)

                ret, thresh = cv2.threshold(single_cnt_mask, 1, 255, cv2.THRESH_BINARY)
                new_mask, contours, hierarchy = cv2.findContours(
                    thresh,
                    cv2.RETR_CCOMP,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                new_good_contours_small.append(contours[0])

        # merge all candidates
        all_contours = new_good_contours_small + \
            new_good_contours_large + \
            good_contours_color_final

        border_contours, non_border_contours = utils_extra.find_border_contours(
            all_contours,
            img_s.shape[0],
            img_s.shape[1]
        )

        # fill the border contours
        filled_border_contours = []

        for c in border_contours:
            filled_mask = utils_extra.fill_border_contour(c, img_s.shape)

            new_mask, contours, hierarchy = cv2.findContours(
                filled_mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE
            )

            filled_border_contours.append(contours[0])

        # recombine the filled border contours and non-border contours
        all_contours = filled_border_contours + non_border_contours

        # now that we have all our contour candidates, we need to check for overlaps
        # using bounding boxes
        all_rects = []

        for c in all_contours:
            all_rects.append(cv2.boundingRect(c))

        rect_combos = []

        for i, r1 in enumerate(all_rects[:-1]):
            a1 = r1[2] * r1[3]

            for j, r2 in enumerate(all_rects[i + 1:]):
                max_x0 = max([r1[0], r2[0]])
                min_x1 = min([r1[0] + r1[2], r2[0] + r2[2]])

                if min_x1 < max_x0:
                    continue

                max_y0 = max([r1[1], r2[1]])
                min_y1 = min([r1[1] + r1[3], r2[1] + r2[3]])

                if min_y1 < max_y0:
                    continue

                a_int = (min_x1 - max_x0) * (min_y1 - max_y0)

                if a_int >= 0.7 * a1:
                    a2 = r2[2] * r2[3]

                    if a_int > 0.7 * a2:
                        rect_combos.append({i, i + j + 1})

        i = 0

        while i < len(rect_combos[:-1]):
            j = i + 1

            while j < len(rect_combos):
                if len(rect_combos[i].intersection(rect_combos[j])) > 0:
                    rect_combos[i] = rect_combos[i].union(rect_combos[j])
                    rect_combos.remove(rect_combos[j])
                    j -= 1
                j += 1
            i += 1

        c_combo_list = []

        for c in rect_combos:
            c_combo_list.extend(list(c))

        combined_contour_masks = []

        for c in rect_combos:
            c_idx_list = list(c)
            c_list = []

            for ci in c_idx_list:
                c_list.append(all_contours[ci])

            combined_contour_masks.append(utils_extra.find_contour_union(c_list, img_s.shape))

        for i, c in enumerate(all_contours):
            if i in c_combo_list:
                continue

            c_mask = np.zeros(img_s.shape, dtype=np.uint8)
            cv2.drawContours(c_mask, [c], 0, 255, cv2.FILLED)
            combined_contour_masks.append(c_mask)

        model_classes = list(self.pipe.named_steps['classification'].classes_)
        prob_threshold = 2 * (1.0 / len(model_classes))
        final_contours = []

        for m in combined_contour_masks:
            new_mask, contours, hierarchy = cv2.findContours(
                m,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE
            )
            c = contours[0]

            features = utils.generate_features(img, c)
            features_df = pd.DataFrame([features])
            probabilities = self.pipe.predict_proba(features_df.drop('label', axis=1))

            if probabilities.max() < prob_threshold:
                continue

            final_contours.append(
                {
                    'contour': c,
                    'prob': {a: probabilities[0][i] for i, a in enumerate(model_classes)}
                }
            )

        self.test_results = final_contours
