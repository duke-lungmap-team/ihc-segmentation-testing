from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy

def obtain_data_from_dataset(dataset, input_shape, output_shape):
    imgs, masks = dataset.load_all_data_into_memory()
    assert masks.shape[3]==output_shape[2]
    new_image_shape = input_shape
    new_shape = output_shape
    images_resize = np.zeros(shape=(imgs.shape[0],) + new_image_shape)
    for ind in range(imgs.shape[0]):
        images_resize[ind,:,:,:] = scipy.misc.imresize(imgs[ind], new_image_shape)

    masks_resize = np.zeros(shape=(masks.shape[0],) + new_shape)
    for idx in range(masks_resize.shape[0]):
        for class_idx in range(masks_resize.shape[3]):
            masks_resize[idx,:,:,class_idx] = scipy.misc.imresize(masks[idx,:,:,class_idx], new_shape)
    return images_resize, masks_resize, dataset.class_names[1:]


def get_generator(images_resize, masks_resize):
    #TODO: the two generators must have same attributes or the seed get's messed up and disconnected from one another
    image_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        # zoom_range=[0.5, 1.0]
    )
    mask_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        # zoom_range=[0.5, 1.0]
    )

    seed = 1
    image_datagen.fit(images_resize, augment=True, seed=seed)
    mask_datagen.fit(masks_resize, augment=True, seed=seed)

    image_generator = image_datagen.flow(
        images_resize,
        batch_size=1,
        seed=seed
    )

    mask_generator = mask_datagen.flow(
        masks_resize,
        batch_size=1,
        seed=seed
    )

    generator = zip(image_generator, mask_generator)
    return generator

def unet_generators(dataset, input_shape=(572, 572, 3), output_shape=(570, 570, 5)):
    imgs_train, masks_train, class_names = obtain_data_from_dataset(dataset, input_shape, output_shape)
    return get_generator(imgs_train, masks_train), class_names


