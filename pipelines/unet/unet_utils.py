import numpy as np
import skimage


def load_all_data_into_memory(dataset_object):
    height = dataset_object.image_info[0]['height']
    width = dataset_object.image_info[0]['width']
    imgs = np.zeros(
        [dataset_object.image_ids.shape[0], height, width, 3],
        dtype=np.uint8
    )
    masks = np.zeros(
        [dataset_object.image_ids.shape[0], height, width, len(dataset_object.class_names[1:])],
        dtype=np.uint8
    )
    for i, imgid in enumerate(dataset_object.image_ids):
        imgs[i, :, :, :] = dataset_object.load_image(imgid)
        masks[i, :, :, :] = load_mask_unet(dataset_object, imgid)
    return imgs, masks

def load_mask_unet(dataset_object, image_id):
    """Generate instance masks for shapes of the given image ID.
    """
    info = dataset_object.image_info[image_id]
    names = dataset_object.class_names[1:]
    mask = np.zeros([info['height'], info['width'], len(names)], dtype=np.uint8)
    for polygon in info['polygons']:
        position = names.index(polygon['label'])
        y = polygon['points'][:, 1]
        x = polygon['points'][:, 0]
        rr, cc = skimage.draw.polygon(y, x)
        mask[rr, cc, position] = 1
    return mask


def load_all_data_into_memory_w_mask_bounds_channel(dataset_object):
    height = dataset_object.image_info[0]['height']
    width = dataset_object.image_info[0]['width']
    imgs = np.zeros([dataset_object.image_ids.shape[0], height, width, 3], dtype=np.uint8)
    masks = np.zeros([dataset_object.image_ids.shape[0], height, width, 2], dtype=np.uint8)
    for i, imgid in enumerate(dataset_object.image_ids):
        imgs[i, :, :, :] = dataset_object.load_image(imgid)
        masks[i, :, :, 0] = dataset_object.load_mask_unet(imgid).squeeze()
        masks[i, :, :, 1] = dataset_object.load_bounds_unet(imgid).squeeze()
    return imgs, masks
