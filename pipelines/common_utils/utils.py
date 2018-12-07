import json
from PIL import Image
import numpy as np
import pandas as pd
from operator import itemgetter
import os
from matplotlib import patches
import matplotlib.pyplot as plt

# this is just to un-confuse pycharm
try:
    from cv2 import cv2
except ImportError:
    import cv2


def get_training_data_for_image_set(image_set_dir):
    # Each image set directory will have a 'regions.json' file. This regions file
    # has keys of the image file names in the image set, and the value for each image
    # is a dict of class labels, and the value of those labels is a list of
    # segmented polygon regions.
    # First, we will read in this file and get the file names for our images
    regions_file = open(os.path.join(image_set_dir, 'regions.json'))
    regions_json = json.load(regions_file)
    regions_file.close()

    # output will be a dictionary of training data, were the polygon points dict
    # is a numpy array. The keys will still be the image names
    training_data = {}

    for image_name, regions_dict in regions_json.items():
        tmp_image = Image.open(os.path.join(image_set_dir, image_name))
        tmp_image = np.asarray(tmp_image)

        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_RGB2HSV)

        training_data[image_name] = {
            'hsv_img': tmp_image,
            'regions': []
        }

        for label, regions in regions_dict.items():

            for region in regions:
                points = np.empty((0, 2), dtype='int')

                for point in region:
                    points = np.append(points, [[point[0], point[1]]], axis=0)

                training_data[image_name]['regions'].append(
                    {
                        'label': label,
                        'points': points
                    }
                )

    return training_data


def compute_bbox(contour):
    x1, y1, w, h = cv2.boundingRect(contour)

    return [x1, y1, x1 + w, y1 + h]


def do_boxes_overlap(box1, box2):
    # if the maximum of both boxes left corner is greater than the
    # minimum of both boxes right corner, the boxes cannot overlap
    max_x_left = max([box1[0], box2[0]])
    min_x_right = min([box1[2], box2[2]])

    if min_x_right < max_x_left:
        return False

    # Likewise for the y-coordinates
    max_y_top = max([box1[1], box2[1]])
    min_y_bottom = min([box1[3], box2[3]])

    if min_y_bottom < max_y_top:
        return False

    return True


def make_boolean_mask(contour, img_dims):
    mask = np.zeros(img_dims, dtype=np.uint8)
    cv2.drawContours(
        mask,
        [contour],
        0,
        255,
        cv2.FILLED
    )

    # return boolean array
    return mask > 0


def make_binary_mask(contour, img_dims):
    mask = np.zeros(img_dims, dtype=np.uint8)
    cv2.drawContours(
        mask,
        [contour],
        0,
        1,
        cv2.FILLED
    )

    # return boolean array
    return mask


def find_overlapping_regions(true_regions, test_regions):
    true_boxes = []
    true_classes = []
    test_boxes = []
    test_classes = []
    test_scores = []

    img_dims = true_regions['hsv_img'].shape[:2]

    for r in true_regions['regions']:
        true_boxes.append(compute_bbox(r['points']))
        true_classes.append(r['label'])

    for r in test_regions:
        test_boxes.append(compute_bbox(r['contour']))

        max_prob = max(r['prob'].items(), key=itemgetter(1))

        test_classes.append(max_prob[0])
        test_scores.append(max_prob[1])

    # now we are ready to find the overlaps, we'll keep track of them with a dictionary
    # where the keys are the true region's index. The values will be a dictionary of
    # overlapping test regions, organized into 2 groups:
    #   - matching overlaps: overlaps where the test & true region labels agree
    #   - non-matching overlaps: overlaps where the test & true region labels differ
    #
    # Each one of those groups will be keys with a value of another list of dictionaries,
    # with the overlapping test region index along with the IoU value.
    # There are 2 other cases to cover:
    #   - true regions with no matching overlaps (i.e. missed regions)
    #   - test regions with no matching overlaps (i.e. false positives)
    overlaps = {}
    true_match_set = set()
    test_match_set = set()

    for i, r1 in enumerate(true_boxes):
        true_mask = None  # reset to None, will compute as needed

        for j, r2 in enumerate(test_boxes):
            if not do_boxes_overlap(r1, r2):
                continue

            # So you're saying there's a chance?
            # If we get here, there is a chance for an overlap but it is not guaranteed,
            # we'll need to check the contours' pixels
            if true_mask is None:
                # we've never tested against this contour yet, so render it
                true_mask = make_boolean_mask(true_regions['regions'][i]['points'], img_dims)

            # and render the test contour
            test_mask = make_boolean_mask(test_regions[j]['contour'], img_dims)

            intersect_mask = np.bitwise_and(true_mask, test_mask)
            intersect_area = intersect_mask.sum()

            if not intersect_area > 0:
                # the bounding boxes overlapped, but the contours didn't, skip it
                continue

            union_mask = np.bitwise_or(true_mask, test_mask)
            true_match_set.add(i)
            test_match_set.add(j)

            if i not in overlaps:
                overlaps[i] = {
                    'true_label': true_classes[i],
                    'true': [],
                    'false': []
                }

            test_result = {
                'test_index': j,
                'iou': intersect_area / union_mask.sum()
            }

            if true_classes[i] == test_classes[j]:
                overlaps[i]['true'].append(test_result)
            else:
                overlaps[i]['false'].append(test_result)

    missed_regions = true_match_set.symmetric_difference((range(0, len(true_boxes))))
    false_positives = test_match_set.symmetric_difference((range(0, len(test_boxes))))

    return {
        'overlaps': overlaps,
        'missed_regions': missed_regions,
        'false_positives': false_positives
    }

    
def calc_recall(tps, fns):
    eps = np.spacing(1)
    recall = tps / (tps + fns + eps)
    return recall


def calc_precision(tps, fps):
    eps = np.spacing(1)
    precision = tps / (tps + fps + eps)
    return precision


def generate_iou_pred_matrices(true_regions, test_regions):
    true_boxes = []
    test_boxes = []
    img_dims = true_regions['hsv_img'].shape[:2]

    for r in true_regions['regions']:
        true_boxes.append(compute_bbox(r['points']))

    for r in test_regions:
        test_boxes.append(compute_bbox(r['contour']))

    iou_mat = np.zeros((len(test_boxes), len(true_boxes)))
    pred_mat = iou_mat.copy()

    for i, r1 in enumerate(true_boxes):
        true_mask = None  # reset to None, will compute as needed

        for j, r2 in enumerate(test_boxes):
            if not do_boxes_overlap(r1, r2):
                continue

            # So you're saying there's a chance?
            # If we get here, there is a chance for an overlap but it is not guaranteed,
            # we'll need to check the contours' pixels
            if true_mask is None:
                # we've never tested against this contour yet, so render it
                true_mask = make_boolean_mask(true_regions['regions'][i]['points'], img_dims)

            # and render the test contour
            test_mask = make_boolean_mask(test_regions[j]['contour'], img_dims)

            intersect_mask = np.bitwise_and(true_mask, test_mask)
            intersect_area = intersect_mask.sum()

            if not intersect_area > 0:
                # the bounding boxes overlapped, but the contours didn't, skip it
                continue

            union_mask = np.bitwise_or(true_mask, test_mask)
            iou = intersect_area / union_mask.sum()
            iou_mat[j, i] = iou
            types, value = max(test_regions[j]['prob'].items(), key=itemgetter(1))
            if types == true_regions['regions'][i]['label']:
                pred_mat[j, i] = value
    return iou_mat, pred_mat


def generate_tp_fn_fp(iou_mat, pred_mat, iou_thresh=0.33, pred_thresh=0.25):
    tp = {}
    for i in reversed(list(np.argsort(pred_mat, axis=None))):
        predind, gtind = np.unravel_index(i, pred_mat.shape)
        if iou_mat[predind, gtind] > iou_thresh:
            # TODO optionally only add if the prediction isn't already in tp.values()?
            if pred_mat[predind, gtind] > pred_thresh:
                tp[gtind] = predind
    fn = set(range(iou_mat.shape[1])) - set(tp.keys())
    fp = set(range(iou_mat.shape[0])) - set(tp.values())
    return tp, fn, fp


def generate_dataframe_aggregation_tp_fn_fp(
        true_regions,
        test_regions,
        iou_mat,
        pred_mat,
        tp,
        fn,
        fp
):
    class_names = set()
    for x in true_regions['regions']:
        class_names.add(x['label'])
    for x in test_regions:
        c, value = max(x['prob'].items(), key=itemgetter(1))
        class_names.add(c)
    results = {k: {'tp': [], 'fp': [], 'fn': []} for k in class_names}
    df = pd.DataFrame({'category': list(class_names)})
    df['TP'] = [0 for _ in list(class_names)]
    df['FP'] = [0 for _ in list(class_names)]
    df['FN'] = [0 for _ in list(class_names)]
    df['GTc'] = [0 for _ in list(class_names)]
    for x in true_regions['regions']:
        c = x['label']
        mask = (df['category'] == c)
        df.loc[mask, 'GTc'] = df.loc[mask, 'GTc'] + 1
    for x in tp.items():
        save = {
            'iou': iou_mat[x[1], x[0]],
            'prob': pred_mat[x[1], x[0]],
            'test_ind': x[1]
        }
        c = true_regions['regions'][x[0]]['label']
        results[c]['tp'].append(save)
        results[c]['tp'].append({'gt_ind': x[0]})
        mask = (df['category'] == c)
        df.loc[mask, 'TP'] = df.loc[mask, 'TP'] + 1
    for x in fn:
        c = true_regions['regions'][x]['label']
        results[c]['fn'].append({'gt_ind': x})
        mask = (df['category'] == c)
        df.loc[mask, 'FN'] = df.loc[mask, 'FN'] + 1
    for x in fp:
        c, value = max(test_regions[x]['prob'].items(), key=itemgetter(1))
        save = {
            'iou': np.max(iou_mat[x, :]),
            'test_ind': x
        }
        results[c]['fp'].append(save)
        mask = (df['category'] == c)
        df.loc[mask, 'FP'] = df.loc[mask, 'FP'] + 1
    df['precision'] = df.apply(lambda row: calc_precision(row['TP'], row['FP']), axis=1)
    df['recall'] = df.apply(lambda row: calc_recall(row['TP'], row['FN']), axis=1)
    return df, results


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_class_prediction_overlaps(
        image,
        segments,
        true_regions,
        test_regions,
        figsize=(16, 16),
        show_mask=True,
        show_bbox=True
):
    for key, x in segments.items():
        # If no axis is passed, create one and automatically call show()
        ax = None
        if not ax:
            _, ax = plt.subplots(1, figsize=figsize)
            auto_show = True
        else:
            auto_show = False

        # Generate random colors
        # Number of color segments (choosing three to match tp, fp, fn
        # colors = colors or random_colors(3)
        colors = [(0.0, 1.0, 0.40000000000000036),
                  (1.0, 0.0, 1.0),
                  (1.0, 1.0, 0.0)]
        color_labels = ['tp', 'fn', 'fp']
        # Show area outside image boundaries.
        height, width = image.shape[:2]
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ax.set_title(key)
        masked_image = image.astype(np.uint32).copy()
        for typekey, t in x.items():
            color = colors[color_labels.index(typekey)]
            for seg in t:
                if 'gt_ind' in list(seg.keys()):
                    contour = true_regions['regions'][seg['gt_ind']]['points']
                    seglabel = 'gt'
                elif 'test_ind' in list(seg.keys()):
                    contour = test_regions[seg['test_ind']]['contour']
                    if 'prob' in list(seg.keys()):
                        seglabel = 'IOU: {0:.2}, PROB: {0:.2%}'.format(seg['iou'], seg['prob'])
                    else:
                        seglabel = 'IOU: {0:.2}'.format(seg['iou'])
                else:
                    continue

                x1, y1, x2, y2 = compute_bbox(contour)
                if show_bbox:
                    p = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=0.7,
                        linestyle="dashed",
                        edgecolor=color,
                        facecolor='none'
                    )
                    ax.add_patch(p)
                ax.text(
                    x1,
                    y1 + 8,
                    seglabel,
                    color='w',
                    size=15,
                    backgroundcolor="none"
                )
                # Mask
                mask = make_binary_mask(contour, (masked_image.shape[0], masked_image.shape[1]))
                if show_mask:
                    masked_image = apply_mask(masked_image, mask, color)

        ax.imshow(masked_image.astype(np.uint8))
        if auto_show:
            plt.show()
