from pipelines.base_pipeline import Pipeline
from pipelines.unet.unet_architecture_paper import unet_padded
from pipelines.common_utils.lungmap_dataset import LungmapDataSet
from pipelines.unet.data_pipeline import unet_generators
from keras.callbacks import ModelCheckpoint
import os
from keras.models import load_model

class LungmapUnetPipeline(Pipeline):
    def __init__(self, image_set_dir, model_dir='tmp/models/unet', test_img_index=0):
        super(LungmapUnetPipeline, self).__init__(image_set_dir, test_img_index)

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

        self.train_generator, self.class_names = unet_generators(self.dataset_train)
        self.val_generator, _ = unet_generators(self.dataset_validation)
        self.model = unet_padded(len(self.class_names))

    def train(self, epochparam=5):
        model_checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, 'unet_lungmap.hdf5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True)
        self.model.fit_generator(
            self.train_generator,
            validation_data = self.val_generator,
            validation_steps=5,
            steps_per_epoch=30,
            epochs=epochparam,
            callbacks=[model_checkpoint],
        )
    def test(self, image, model_save_dir='models'):
        self.trained_model = load_model(os.path.join(model_save_dir, 'unet_lungmap.hdf5'))
        return self.trained_model.predict(image)

# up = UnetPipeline('/Users/nn31/Documents/ihc-segmentation-testing/data/image_set_73')

# class ShapesUnetPipeline:
#     def __init__(self, number_of_shapes=10, number_of_images=4, test_img_index=0):
#         dataset_train = ShapesDataset(number_of_shapes, iou_threshold=0.001)
#         dataset_train.load_shapes(number_of_images, 2475, 2475)
#         dataset_train.prepare()
#         self.train_generator, self.test_generator, self.class_names = return_generators(dataset_train, [0])
#         self.model = unet(len(self.class_names))
#     def train(self, model_save_dir='models', epochparam=5):
#         model_checkpoint = ModelCheckpoint(
#             os.path.join(model_save_dir, 'unet_shapes.hdf5'),
#             monitor='loss',
#             verbose=1,
#             save_best_only=True)
#         self.model.fit_generator(
#             self.train_generator,
#             steps_per_epoch=3,
#             epochs=epochparam,
#             callbacks=[model_checkpoint],
#         )
#
#
# sup = ShapesUnetPipeline()