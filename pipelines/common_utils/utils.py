import json
import PIL
import numpy as np
import cv2
from operator import itemgetter
import os
import paramiko


def get_training_data_for_image_set(image_set_dir):
    # Each image set directory will have a 'regions.json' file. This regions file
    # has keys of the image file names in the image set, and the value for each image
    # is a list of segmented polygon regions.
    # First, we will read in this file and get the file names for our images
    regions_file = open(os.path.join(image_set_dir, 'regions.json'))
    regions_json = json.load(regions_file)
    regions_file.close()

    # output will be a dictionary of training data, were the polygon points dict is a
    # numpy array. The keys will still be the image names
    training_data = {}

    for image_name, sub_regions in regions_json.items():
        # noinspection PyUnresolvedReferences
        tmp_image = PIL.Image.open(os.path.join(image_set_dir, image_name))
        tmp_image = np.asarray(tmp_image)

        # noinspection PyUnresolvedReferences
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_RGB2HSV)

        training_data[image_name] = {
            'hsv_img': tmp_image,
            'regions': []
        }

        for region in sub_regions:
            points = np.empty((0, 2), dtype='int')

            for point in sorted(region['points'], key=itemgetter('order')):
                points = np.append(points, [[point['x'], point['y']]], axis=0)

            training_data[image_name]['regions'].append(
                {
                    'label': region['anatomy'],
                    'points': points
                }
            )

    return training_data


def compute_bbox(contour):
    # noinspection PyUnresolvedReferences
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
    # noinspection PyUnresolvedReferences
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
    # noinspection PyUnresolvedReferences
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


def put_file_to_remote(model_name, file):
    """
    Put a file to our remote store for sharing
    :param model_name: str: name of model, this is used to stash file in particular folder
    :param file: the unc path of the file to put to remote
    :return:
    """
    k = paramiko.RSAKey.from_private_key_file(
        os.getenv('pem_file')
    )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        hostname=os.getenv('remote_file_server'),
        username=os.getenv('username'),
        pkey=k
    )
    remote_file_path = os.path.join(
        '/home',
        os.getenv('username'),
        'ihc-segmentation-testing-models',
        model_name,
        os.path.basename(file)
    )
    ftp_client = c.open_sftp()
    ftp_client.put(file, remote_file_path)
    ftp_client.close()
    c.close()


def get_file_from_remote(model_name, file_name):
    k = paramiko.RSAKey.from_private_key_file(
        os.getenv('pem_file')
    )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        hostname=os.getenv('remote_file_server'),
        username=os.getenv('username'),
        pkey=k
    )
    remote_file_path = os.path.join(
        '/home',
        os.getenv('username'),
        'ihc-segmentation-testing-models',
        model_name,
        os.path.basename(file_name)
    )
    local_file_path = os.path.join(
        'tmp',
        'models',
        model_name,
        file_name
    )
    ftp_client = c.open_sftp()
    ftp_client.get(remote_file_path, local_file_path)
    ftp_client.close()
    c.close()
