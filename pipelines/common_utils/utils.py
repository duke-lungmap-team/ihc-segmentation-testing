import json
import os
import PIL
import numpy as np
import cv2
from operator import itemgetter


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
