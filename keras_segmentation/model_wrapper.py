import glob
import os
import random
import warnings

import cv2
import keras
import numpy as np
from tqdm import tqdm

from .data_utils.data_loader import image_segmentation_generator, verify_segmentation_dataset, get_image_arr
from .models import model_from_name
from .models.config import IMAGE_ORDERING

random.seed(0)
class_colours = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5000)]


class WrappedModel:
    """WrappedModel class wraps a provided keras model and provide an interface to simplify use.

    Attributes:
        model: The keras model that is wrapped
        curr_epoch: An integer keeping track of number of epochs that have been trained.
    """

    def __init__(self, keras_model, n_classes, input_height=None, input_width=None):
        """Inits a WrappedModel object with passed args.

        Args:
            keras_model: (str or Keras.Model) Either name of model found in keras_segmentation/models/__init__.py or a
                function handle/class handle.
            n_classes (int): Number of output classes
            input_height (int): height of input images.
            input_width (int): width of input images.
        """

        # ensure n_classes value is valid
        assert (type(n_classes) is int) and n_classes > 0, "n_classes must be an integer value greater than 0."

        # call model constructor depending on how model is specified
        if type(keras_model) is str:
            model_constructor = model_from_name[keras_model]
        elif callable(keras_model):
            model_constructor = keras_model
        else:
            raise AssertionError("Enter a modelname found in keras_segmentation/models/__init__.py or manually pass"
                                 " the function handle or class handle to a model found in the"
                                 " keras_segmentation/models directory.")

        # pass through input_height and input_width if specified, use default size if not
        if (input_height is None) and (input_width is None):
            self.model = model_constructor(n_classes=n_classes)
        else:
            self.model = model_constructor(n_classes=n_classes, input_height=input_height, input_width=input_width)

        # set curr_epoch at 0 when new model initialized
        self.curr_epoch = 0

        # remove Methods dynamically attached to model in get_segmentation_model function under
        # keras_segmentation/models/model_utils
        del self.model.train
        del self.model.predict_segmentation
        del self.model.predict_multiple
        del self.model.evaluate_segmentation


    def train_model(self, train_images, train_annotations, epochs=5, batch_size=2, checkpoints_path=None,
                    resume_training=True, validate=False, val_images=None, val_annotations=None, verify_dataset=True,
                    steps_per_epoch=512, optimizer_name="adadelta"):
        """Method to train the current model.

        Args:
            train_images: (str or ndarray) Path to train images or ndarray of already read images.
            train_annotations: (str or ndarray) Path to train annotations or ndarray of already read annotations.
            epochs: (int) Number of epochs to train.
            batch_size: (int) Number of images in batch.
            checkpoints_path: (str) Path to folder for model checkpointing.
            resume_training: (bool) Whether or not to start checkpointing at epoch 0 or continue from curr_epoch.
            validate: (bool) Whether or not to evaluate performance on a validation set after each epoch.
            val_images: (str or ndarray) Path to validation images or ndarray of already read images.
            val_annotations: (str or ndarray) Path to validation annotations or ndarray of already read annotations.
            verify_dataset: (bool) Whether or not to verify provided images and annotations.
            steps_per_epoch: (int): Number of images to train over in one epoch.
            optimizer_name: (str): Optimizer to use for training.

        Returns:
            A Keras History object. Metrics from training can be accessed through the History.history dictionary.
        """

        if verify_dataset:
            verify_segmentation_dataset(train_images, train_annotations, self.model.n_classes)

        # get generator for training images
        train_gen = image_segmentation_generator(train_images, train_annotations, batch_size, self.model.n_classes,
                                                 self.model.input_height, self.model.input_width,
                                                 self.model.output_height, self.model.output_width)

        if validate:
            assert val_images is not None and val_annotations is not None, "Validate specified, but val_images and" \
                                                                           " val_annotations not provided."

            if verify_dataset:
                verify_segmentation_dataset(val_images, val_annotations, self.model.n_classes)

            # get generator for validation images
            val_gen = image_segmentation_generator(val_images, val_annotations, batch_size, self.model.n_classes,
                                                   self.model.input_height, self.model.input_width,
                                                   self.model.output_height, self.model.output_width)

        callbacks = []

        if checkpoints_path is not None:
            # checkpoint file prefix
            checkpoint_file = "{checkpoints_path}/saved_weights".format(checkpoints_path=checkpoints_path)

            # check if checkpoint directory already exists, create if not
            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
            else:
                files = glob.glob(checkpoint_file + "-*")

                # if checkpoint files already exist in directory, raise warning of possibility of overwriting.
                if len(files) > 0:
                    warnings.warn("Provided checkpoints_path already has existing checkpoints. Make sure"
                                  " resume_training=True to continue training. Proceeding but previous"
                                  " checkpoints may be overwritten")

            # add suffix to checkpoint file
            path_template = checkpoint_file + "-epoch_{epoch:02d}.hdf5"
            monitor_metric = "val_acc" if validate else "acc"

            # Add ModelCheckpoint callback to callbacks list.
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=path_template,
                    monitor=monitor_metric,
                    save_weights_only=True,
                    verbose=1,
                )
            )

        # Compile the training graph
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])

        # set initial epoch for checkpoint purposes.
        initial_epoch = self.curr_epoch if resume_training else 0

        if initial_epoch >= epochs:
            raise AssertionError("Please increase epochs value, training is starting at epoch {} and total number of"
                                 " epochs specified for training is {}".format(initial_epoch, epochs))

        # Start model training
        history = self.model.fit_generator(
            generator=train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=val_gen if validate else None,
            validation_steps=100 if validate else None,
            shuffle=True,
            use_multiprocessing=True,
            initial_epoch=initial_epoch
        )

        self.curr_epoch = epochs
        return history


    def load_weights(self, saved_weights_path):
        """Load saved weights from specified checkpoint

        Args:
            saved_weights_path: (str) Path to specific checkpoint file
        """
        self.curr_epoch = int(saved_weights_path.split("_")[-1].split(".")[0])
        self.model.load_weights(saved_weights_path)


    def predict(self, input_img, out_fname=None):
        """Perform inference on specified image.

        Args:
            input_img: (str or ndarray) Path to an image, or an already read image passed as a np.ndarray
            out_fname: (str) File to save segmented image as.

        Returns:
            a np.ndarray of the segmented image
        """
        assert (type(input_img) == str) or (type(input_img) == np.ndarray), "Input should be the ndarray of an image" \
                                                                            " or a filename."

        if type(input_img) == str:
            input_img = cv2.imread(input_img)

        # preprocess image
        x = get_image_arr(input_img, self.model.input_width, self.model.input_height, odering=IMAGE_ORDERING)

        # get segmentation prediction
        prediction = self.model.predict(np.array([x]))[0]

        # format prediction
        prediction = prediction.reshape(
            (self.model.output_height, self.model.output_width, self.model.n_classes)
        ).argmax(axis=2)

        # create zeroed ndarray to hold segmented image
        seg_img = np.zeros((self.model.output_height, self.model.output_width, 3))
        colours = class_colours

        # apply class colours to relevant image segments
        for c in range(self.model.n_classes):
            seg_img[:, :, 0] += ((prediction[:, :] == c) * (colours[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((prediction[:, :] == c) * (colours[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((prediction[:, :] == c) * (colours[c][2])).astype('uint8')
        seg_img = cv2.resize(seg_img, (self.model.input_width, self.model.input_height))

        if out_fname is not None:
            cv2.imwrite(out_fname, seg_img)

        return prediction


    def predict_multiple(self, input_imgs=None, input_directory=None, output_directory=None):
        """Perform inference on a set of images.

        Args:
            input_imgs: (ndarray) a list containing images for segmentation
            input_directory: (str) Path to a folder containing images for segmentation
            output_directory: (str) Path to store segmented images

        Returns:
            A list of segmented images.
        """
        # get filenames if input_directory specified
        if input_imgs is None and (input_directory is not None):
            input_imgs = glob.glob(os.path.join(input_directory, "*.jpg")) + \
                          glob.glob(os.path.join(input_directory, "*.png")) + \
                          glob.glob(os.path.join(input_directory, "*.jpeg"))

        assert type(input_imgs) == list

        prediction_list = []

        if output_directory is None:
            # list comprehension if segmented images not being saved
            prediction_list = [self.predict(input_img, None) for _, input_img in enumerate(tqdm(input_imgs))]
        else:
            for i, input_img in enumerate(tqdm(input_imgs)):

                # if img is path to file, use filename for segmented image in out_directory. Else use iteration
                if type(input_img) == str:
                    out_fname = os.path.join(output_directory, os.path.basename(input_img))
                else:
                    out_fname = os.path.join(output_directory, str(i) + ".jpg")

                prediction_list.append(self.predict(input_img, out_fname))

        return prediction_list
