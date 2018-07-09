from pipelines.common_utils.dataset import DataSet
import numpy as np
import skimage.draw
import cv2


class LungmapDataSet(DataSet):
    def __init__(self, test_image_indices=[0]):
        super(LungmapDataSet, self).__init__()
        self.test_image_indices = test_image_indices

    def load_data_set(self, data_set):
        """
        Load a data set.
        data_set: Dictionary of data, keys are image names, values are sub-regions
        """
        anatomy = set()

        for img_name, img_data in data_set.items():
            self.add_image(
                "lungmap",
                image_id=img_name,  # use file name as a unique image id
                img=cv2.cvtColor(img_data['hsv_img'], cv2.COLOR_HSV2RGB),
                width=img_data['hsv_img'].shape[0],
                height=img_data['hsv_img'].shape[1],
                polygons=img_data['regions']
            )
            for region in img_data['regions']:
                anatomy.add(region['label'])

        anatomy = list(anatomy)  # TODO: should this be sorted to ensure order?
        for i, a in enumerate(anatomy):
            self.add_class("lungmap", i+1, a)

    @staticmethod
    def bbox2(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

    def load_mask(self, image_id, with_bb=False):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = []
        b_boxes = []
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            y = p['points'][:, 1]
            x = p['points'][:, 0]
            rr, cc = skimage.draw.polygon(y, x)
            mask[rr, cc, i] = 1
            y1, y2, x1, x2 = self.bbox2(mask[:, :, i])
            b_boxes.append([y1, x1, y2, x2])
            my_id = [x['id'] for x in self.class_info if p['label'] == x['name']][0]
            class_ids.append(my_id)
        bboxes = np.vstack(b_boxes)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        if with_bb:
            return mask, np.array(class_ids, dtype=np.int32), bboxes
        else:
            return mask, np.array(class_ids, dtype=np.int32)
