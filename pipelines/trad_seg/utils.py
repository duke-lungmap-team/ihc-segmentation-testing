import numpy as np
from cv2_extras import utils as cv2x
from cv_color_features import utils as color_utils

# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2

# Dilation/erosion kernels
cross_strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
block_strel = np.ones((3, 3))
ellipse_strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ellipse90_strel = np.rot90(ellipse_strel)
circle_strel = np.bitwise_or(ellipse_strel, ellipse90_strel)


def process_image(img_hsv):
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    img_shape = (img_hsv.shape[0], img_hsv.shape[1])

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

    contours = cv2x.filter_contours_by_size(edge_mask, min_size=15 * 15)

    edge_mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(edge_mask, contours, -1, 255, -1)

    edge_mask = cv2.dilate(edge_mask, cross_strel, iterations=1)

    return b_suppress_img, edge_mask


def generate_color_contours(img_hsv, edge_mask):
    img_shape = (img_hsv.shape[0], img_hsv.shape[1])

    tmp_color_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    tmp_color_img = cv2.blur(tmp_color_img, (9, 9))
    tmp_color_img = cv2.cvtColor(tmp_color_img, cv2.COLOR_RGB2HSV)

    color_mask = color_utils.create_mask(
        tmp_color_img,
        [
            'green', 'cyan', 'red', 'violet', 'yellow'
        ]
    )

    # Next, clean up any "noise", say, ~ 11 x 11
    contours = cv2x.filter_contours_by_size(color_mask, min_size=5 * 5)

    color_mask2 = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(color_mask2, contours, -1, 255, -1)

    # do a couple dilation iterations to smooth some outside edges & connect any
    # adjacent cells each other
    color_mask3 = cv2.dilate(color_mask2, cross_strel, iterations=2)
    color_mask3 = cv2.erode(color_mask3, cross_strel, iterations=3)

    # filter out any remaining contours that are the size of roughly 2 cells
    contours = cv2x.filter_contours_by_size(color_mask3, min_size=2 * 40 * 40)

    color_mask3 = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(color_mask3, contours, -1, 255, -1)

    good_color_contours = []

    for i, c in enumerate(contours):
        filled_c_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(filled_c_mask, [c], -1, 255, cv2.FILLED)

        new_mask, signal, orig = cv2x.find_border_by_mask(
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

    return final_color_contours


def generate_saturation_contours(
        img_s,
        blur_kernel,
        mask,
        edge_mask,
        min_size,
        max_size,
        erode_iter,
        max_dilate_percentage,
        blur_type='gaussian'
):
    img_shape = img_s.shape
    if blur_type == 'gaussian':
        img_s_blur = cv2.GaussianBlur(img_s, blur_kernel, 0, 0)
    elif blur_type == 'bilateral':
        img_s_blur = cv2.bilateralFilter(
            img_s,
            blur_kernel[0],
            blur_kernel[0],
            blur_kernel[0]
        )
    else:
        raise ValueError("%s isn't a supported blur type" % blur_type)

    med = np.median(img_s_blur[img_s_blur > 0])

    img_s_blur = cv2.bitwise_and(
        img_s_blur,
        img_s_blur,
        mask=mask
    )
    mode_s = cv2.inRange(img_s_blur, med, 255)
    mode_s = cv2.erode(mode_s, block_strel, iterations=erode_iter)

    good_contours = cv2x.filter_contours_by_size(
        mode_s,
        min_size=min_size,
        max_size=max_size
    )

    # update the signal mask by removing the previous color candidates
    edge_mask = cv2.bitwise_and(edge_mask, edge_mask, mask=mask)
    contours = cv2x.filter_contours_by_size(edge_mask, min_size=21 * 21)
    edge_mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(edge_mask, contours, -1, 255, -1)
    edge_mask = cv2.dilate(edge_mask, (3, 3), iterations=2)

    good_sat_val_contours = []

    for i, c in enumerate(good_contours):
        filled_c_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(filled_c_mask, [c], -1, 255, cv2.FILLED)

        new_mask, signal, orig = cv2x.find_border_by_mask(
            edge_mask,
            filled_c_mask,
            max_dilate_percentage=max_dilate_percentage,
            dilate_iterations=2
        )

        if not orig and signal > 0.7:
            _, contours, _ = cv2.findContours(
                new_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            good_sat_val_contours.append(contours[0])
        elif signal > 0.7:
            good_sat_val_contours.append(c)
        else:
            pass  # ignore contour

    # The previous contours represent the final contour count,
    # but not their final shape/size. The final step is to use approxPolyDP, followed
    # by a few more dilations, but we won't re-draw and extract the contours, as that
    # would possibly connect some of them together. However, we will draw them for
    # the mask to exclude these regions from the next group of candidates
    final_sat_val_contours = []

    for c in good_sat_val_contours:
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

        final_sat_val_contours.append(tmp_contours[0])

    return final_sat_val_contours, edge_mask
