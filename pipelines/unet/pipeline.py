from pipelines.base_pipeline import BasePipeline
from pipelines.unet.unet_architecture_paper import color_model
from pipelines.common_utils.lungmap_dataset import LungmapDataSet
from pipelines.unet.data_pipeline import unet_generators
from keras.callbacks import ModelCheckpoint
import os
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


class LungmapUnetPipeline(BasePipeline):
    def __init__(self,
                 image_set_dir,
                 model_dir='tmp/models/unet',
                 model_name='thing_unet.hdf5',
                 train_images=None,
                 test_images=None,
                 val_images=None,
                 input_shape=(1024, 1024, 3),
                 output_shape=(1024, 1024, 5)
                 ):
        super(LungmapUnetPipeline, self).__init__(image_set_dir)

        # TODO: make sure there are enough images in the image set, as this pipeline
        # needs to separate an image for validation, in addition to the reserved test image.
        self.model_name = os.path.join(model_dir, model_name)
        training_data = {x: self.training_data[x] for x in train_images}
        test_data = {x: self.training_data[x] for x in test_images}
        validation_data = {x: self.training_data[x] for x in val_images}

        self.dataset_train = LungmapDataSet()
        self.dataset_train.load_data_set(training_data)
        self.dataset_train.prepare()

        self.dataset_validation = LungmapDataSet()
        self.dataset_validation.load_data_set(validation_data)
        self.dataset_validation.prepare()

        self.dataset_test = LungmapDataSet()
        self.dataset_test.load_data_set(test_data)
        self.dataset_test.prepare()

        self.modify_dataset(self.dataset_train)
        self.modify_dataset(self.dataset_test)
        self.modify_dataset(self.dataset_validation)

        self.train_generator, self.class_names = unet_generators(
            self.dataset_train,
            self.dataset_train.class_names,
            input_shape,
            output_shape,
        )
        self.val_generator, _ = unet_generators(
            self.dataset_validation,
            self.dataset_train.class_names,
            input_shape,
            output_shape
        )

        self.test_generator, _ = unet_generators(
            self.dataset_test,
            self.dataset_train.class_names,
            input_shape,
            output_shape
        )
        self.model = color_model()

    @staticmethod
    def modify_dataset(which_dataset):
        which_dataset.class_names = ['BG', 'thing']
        new_image_info = []
        for img in which_dataset.image_info:
            tmp_poly = img.pop('polygons')
            new_poly = []
            for poly in tmp_poly:
                poly['label'] = 'thing'
                new_poly.append(poly)
            img['polygons'] = new_poly
            new_image_info.append(img)

    def train(self, epochparam=5):
        model_checkpoint = ModelCheckpoint(
            self.model_name,
            monitor='val_loss',
            verbose=1,
            save_best_only=True)
        self.model.fit_generator(
            self.train_generator,
            validation_data=self.val_generator,
            validation_steps=5,
            steps_per_epoch=30,
            epochs=epochparam,
            callbacks=[model_checkpoint],
        )

    def test(self):
        img, m = self.test_generator
        trained_model = load_model(self.model_name)
        pred = trained_model.predict(img)
        pred_thresh = np.where(pred > 0.5, 1, 0)
        print_all_masks(img, pred_thresh)
        return


def print_all_masks(i, m):
    for x in range(m.shape[3]):
        plt.imshow(i[0, :, :])
        plt.imshow(m[0, :, :, x], cmap='jet', alpha=0.5)
        plt.show()
