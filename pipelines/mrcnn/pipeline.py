from pipelines.base_pipeline import Pipeline
from pipelines.mrcnn import model as modellib
from pipelines.mrcnn.config import Config
from pipelines.common_utils.lungmap_dataset import LungmapDataSet
import os


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


class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()


class MrCNNPipeline(Pipeline):
    def __init__(self, image_set_dir, model_dir, test_img_index=0):
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

    def train(self):
        # self.model.load_weights(
        #     'tmp/models',
        #     by_name=True,
        #     exclude=[
        #         "mrcnn_class_logits",
        #         "mrcnn_bbox_fc",
        #         "mrcnn_bbox",
        #         "mrcnn_mask"
        #     ]
        # )
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

    def test(self):
        # TODO: check which color space model.detect needs & convert as needed
        img = self.training_data[self.test_img_name]['hsv_img']
        model = modellib.MaskRCNN(
            mode="inference",
            config=inference_config,
            model_dir=self.model_dir
        )
        model.load_weights(os.path.join(self.model_dir, 'mask_rcnn_lungmap_0002.h5'))
        return model.detect([img], verbose=1)
