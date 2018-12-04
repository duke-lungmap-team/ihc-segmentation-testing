from pipelines.base_pipeline import BasePipeline
from pipelines.unet.unet_architecture_paper import color_model
from pipelines.common_utils.lungmap_dataset import LungmapDataSet
from pipelines.unet.data_pipeline import unet_generators
from keras.callbacks import ModelCheckpoint
import os
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from cv2_extras import utils as cv2x
from cv_color_features import utils as color_utils
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import copy
from joblib import dump, load
from cv2 import findContours
import cv2
import scipy


class LungmapUnetClassifierPipeline(BasePipeline):
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
        super().__init__(image_set_dir)
        self.test_img_name = test_images[0]
        # TODO: make sure there are enough images in the image set, as this pipeline
        # needs to separate an image for validation, in addition to the reserved test image.
        self.model_name = os.path.join(model_dir, model_name)
        self.td = copy.deepcopy(self.training_data)
        training_data = {x: self.td[x] for x in train_images}
        test_data = {x: self.td[x] for x in test_images}
        validation_data = {x: self.td[x] for x in val_images}
        self.test_images = test_images

        classifier = SVC(probability=True, class_weight='balanced')
        self.pipe = Pipeline(
            [
                ('scaler', MinMaxScaler()),
                ('classification', classifier)
            ]
        )

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
            input_shape,
            output_shape,
        )
        self.val_generator, _ = unet_generators(
            self.dataset_validation,
            input_shape,
            output_shape
        )

        self.test_generator, _ = unet_generators(
            self.dataset_test,
            input_shape,
            output_shape
        )
        self.model = color_model()
        self.training_data_processed = None
        self.test_data_processed = None

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

    def train_classifier(self):
        # process the training data to get custom metrics
        training_data_processed = []

        for img_name, img_data in self.td.items():
            if img_name in self.test_images:
                continue

            non_bg_regions = []

            for region in img_data['regions']:
                training_data_processed.append(
                    color_utils.generate_features(
                        hsv_img_as_numpy=img_data['hsv_img'],
                        polygon_points=region['points'],
                        label=region['label']
                    )
                )

                non_bg_regions.append(region['points'])

            bg_contours = cv2x.generate_background_contours(
                img_data['hsv_img'],
                non_bg_regions
            )

            for region in bg_contours:
                training_data_processed.append(
                    color_utils.generate_features(
                        hsv_img_as_numpy=img_data['hsv_img'],
                        polygon_points=region,
                        label='background'
                    )
                )

        # train the SVM model with the ground truth
        self.training_data_processed = pd.DataFrame(training_data_processed)
        self.training_data_processed.to_csv('tmp/models/unet/classification_data.csv', index=False)
        self.pipe.fit(
            self.training_data_processed.drop('label', axis=1),
            self.training_data_processed['label']
        )
        dump(self.pipe, 'tmp/models/unet/classifier.joblib')

    def test(self):
        img_rgb = cv2.cvtColor(self.training_data[self.test_img_name]['hsv_img'], cv2.COLOR_HSV2RGB)
        img_down = scipy.misc.imresize(img_rgb, (1024, 1024, 3))
        trained_model = load_model(self.model_name)
        classifier = load('tmp/models/unet/classifier.joblib')
        pred = trained_model.predict(img_down.reshape(1, 1024, 1024, 3))
        pred_thresh = np.where(pred > 0.5, 1, 0)
        assert pred_thresh.shape[0] == 1
        pred_thresh_up = scipy.misc.imresize(pred_thresh[0, :, :, 0], (2475, 2475))
        _, contours, _ = findContours(
            pred_thresh_up.astype('uint8'),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contour_filter = [c for c in contours if c.shape[0] > 3]
        hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        test_data_processed = []
        final_contour_results = []
        for c_idx, c in enumerate(contour_filter):
            features = color_utils.generate_features(
                hsv_img,
                c
            )
            features_df = pd.DataFrame([features])
            probabilities = classifier.predict_proba(features_df.drop('label', axis=1))
            labelled_probs = {a: probabilities[0][i] for i, a in enumerate(classifier.classes_)}
            pred_label = max(labelled_probs, key=lambda key: labelled_probs[key])
            pred_label_prob = labelled_probs[pred_label]

            final_contour_results.append(
                {
                    'test_index': c_idx,
                    'contour': c,
                    'prob': labelled_probs
                }
            )

            features['pred_label'] = pred_label
            features['pred_label_prob'] = pred_label_prob
            features['region_file'] = None
            features['region_index'] = c_idx

            test_data_processed.append(features)
        self.test_data_processed = pd.DataFrame(test_data_processed)
        self.test_results = final_contour_results


def print_all_masks(i, m):
    for x in range(m.shape[3]):
        plt.imshow(i[0, :, :])
        plt.imshow(m[0, :, :, x], cmap='jet', alpha=0.5)
        plt.show()
