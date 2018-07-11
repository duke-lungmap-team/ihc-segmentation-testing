from pipelines.base_pipeline import Pipeline
from pipelines.mrcnn import model as modellib
from pipelines.mrcnn.config import Config
from pipelines.common_utils.lungmap_dataset import LungmapDataSet
from pipelines.common_utils.utils import put_file_to_remote, get_file_from_remote
import os
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

class LungMapTrainingConfig(Config):
    """
    Configuration for training
    """
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 50
    # Number of classification classes (including background)
    # TODO: update this to be automatic
    NUM_CLASSES = 6  # Override in sub-classes
    RPN_NMS_THRESHOLD = 0.9
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    USE_MINI_MASK = False
    TRAIN_ROIS_PER_IMAGE = 256
    MAX_GT_INSTANCES = 24
    DETECTION_MAX_INSTANCES = 100


config = LungMapTrainingConfig()


class MrCNNPipeline(Pipeline):
    def __init__(self, image_set_dir, model_dir='tmp/models/maskrcnn', test_img_index=0):
        super(MrCNNPipeline, self).__init__(image_set_dir, test_img_index)

        # TODO: make sure there are enough images in the image set, as this pipeline
        # needs to separate an image for validation, in addition to the reserved test image.
        self.model_dir = model_dir
        training_data = self.training_data.copy()
        test_data = {}
        validation_data = {}
        test_data[self.test_img_name] = training_data.pop(self.test_img_name)
        val_img_name = sorted(training_data.keys())[0]
        validation_data[val_img_name] = training_data.pop(val_img_name)

        self.dataset_train = LungmapDataSet()
        self.dataset_train.load_data_set(training_data)
        self.dataset_train.prepare()

        self.dataset_validation = LungmapDataSet()
        self.dataset_validation.load_data_set(validation_data)
        self.dataset_validation.prepare()

        self.dataset_test = LungmapDataSet()
        self.dataset_test.load_data_set(test_data)
        self.dataset_test.prepare()

        self.model = modellib.MaskRCNN(
            mode="training",
            config=config,
            model_dir=self.model_dir
        )

    def train(self, coco_model_weights='tmp/models/mrcnn/mask_rcnn_coco.h5'):
        """
        Method to train new algorithm
        :param coco_model_weights: required is the pretrained h5 file which contains the coco trained model weights
        :return:
        """
        if not os.path.exists(coco_model_weights):
            logging.info(
                'Could not find pre-trained weights maks_rcnn_coco_h5 in tmp/models/mrcnn, caching now.'
            )
            get_file_from_remote('mrcnn', 'mask_rcnn_coco.h5')
            logging.info(
                'weights now cached: tmp/models/mrcnn/mask_rcnn_coco.h5'
            )

        self.model.load_weights(
            coco_model_weights,
            by_name=True,
            exclude=[
                "mrcnn_class_logits",
                "mrcnn_bbox_fc",
                "mrcnn_bbox",
                "mrcnn_mask"
        ])
        self.model.train(
            self.dataset_train,
            self.dataset_validation,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads'
        )
        self.model.train(
            self.dataset_train,
            self.dataset_validation,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all"
        )

    def test(self, model_weights='tmp/models/mrcnn/mask_rcnn_lungmap_0002.h5'):
        if not os.path.exists(model_weights):
            logging.info(
                'Could not find pre-trained weights mask_rcnn_lungmap_0002.h5 in tmp/models/mrcnn, caching now.'
            )
            get_file_from_remote('mrcnn', 'mask_rcnn_lungmap_0002.h5')
            logging.info(
                'weights now cached: tmp/models/mrcnn/mask_rcnn_lungmap_0002.h5'
            )
        img = self.dataset_test.image_info[0]['img']
        model = modellib.MaskRCNN(
            mode="inference",
            config=config,
            model_dir=self.model_dir
        )
        model.load_weights(
            os.path.join(self.model_dir, 'mask_rcnn_lungmap_0002.h5'),
            by_name=True,
            exclude=[
                "mrcnn_class_logits",
                "mrcnn_bbox_fc",
                "mrcnn_bbox",
                "mrcnn_mask"
            ]
        )
        return model.detect([img], verbose=1)
