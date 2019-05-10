import os
import glob
import warnings

import keras
from .models import model_from_name
from.data_utils.data_loader import image_segmentation_generator, verify_segmentation_dataset


class ModelBase:

    def __init__(self, keras_model, n_classes, input_height=None, input_width=None):
        assert (type(n_classes) is int) and n_classes > 0, "n_classes must be an integer value greater than 0."

        if type(keras_model) is str:
            model_constructor = model_from_name[keras_model]
        elif callable(keras_model):
            model_constructor = keras_model
        else:
            raise AssertionError("Enter a modelname found in keras_segmentation/models/__init__.py or manually pass"
                                 " the function handle to a model found in the keras_segmentation/models directory.")

        if (input_height is None) and (input_width is None):
            self.model = model_constructor(n_classes=n_classes)
        else:
            self.model = model_constructor(n_classes=n_classes, input_height=input_height, input_width=input_width)

        del self.model.train
        del self.model.predict_segmentation
        del self.model.predict_multiple
        del self.model.evaluate_segmentation

    def train_model(self, train_images, train_annotations, epochs=5, batch_size=2, checkpoints_path=None,
                    resume_training=True, validate=False, val_images=None, val_annotations=None, verify_dataset=True,
                    steps_per_epoch=512, optimizer_name="adadelta"):

        if verify_dataset:
            verify_segmentation_dataset(train_images, train_annotations, self.model.n_classes)

        train_gen = image_segmentation_generator(train_images, train_annotations, batch_size, self.model.n_classes,
                                                 self.model.input_height, self.model.input_width,
                                                 self.model.output_height, self.model.output_width)

        if validate:
            assert val_images is not None
            assert val_annotations is not None

            if verify_dataset:
                verify_segmentation_dataset(val_images, val_annotations, self.model.n_classes)

            val_gen = image_segmentation_generator(val_images, val_annotations, batch_size, self.model.n_classes,
                                                   self.model.input_height, self.model.input_width,
                                                   self.model.output_height, self.model.output_width)

        callbacks = []

        initial_epoch = 0

        if checkpoints_path is not None:
            checkpoint_file = "{checkpoints_path}/saved_model".format(checkpoints_path=checkpoints_path)

            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
            else:
                if resume_training:
                    files = glob.glob(checkpoint_file + "-*")

                    initial_epoch = sorted(
                        [int(x.split("-")[1].split("_")[-1]) for x in files],
                        reverse=True
                    )[0]

                else:
                    warnings.warn("Provided checkpoints_path already has existing checkpoints. Make sure"
                                  " resume_training=True to continue training. Proceeding but previous"
                                  " checkpoints may be overwritten")

            if validate:
                path_template = checkpoint_file + "-epoch_{epoch:02d}-valacc_{val_acc:.2f}.hdf5"
                monitor_metric = "val_acc"
            else:
                path_template = checkpoint_file + "-epoch_{epoch:02d}-acc_{acc:.2f}.hdf5"
                monitor_metric = "acc"

            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=path_template,
                    monitor=monitor_metric,
                    verbose=1,
                )
            )

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])

        if validate:
            history = self.model.fit_generator(
                generator=train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks,
                validation_data=val_gen,
                validation_steps=100,
                shuffle=True,
                use_multiprocessing=True,
                initial_epoch=initial_epoch
            )
        else:
            history = self.model.fit_generator(
                generator=train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                initial_epoch=initial_epoch
            )

        return history
