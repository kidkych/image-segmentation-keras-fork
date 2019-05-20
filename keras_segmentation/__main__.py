"""Keras Image Segmentation Fork.

Usage:
    keras_segmentation KERASMODEL train --n_classes=NUM_CLASSES
                (--train_images=TRAIN_IMAGE_PATH --train_annotations=TRAIN_ANNOTATION_PATH)
                [(--validate --val_images=VAL_IMAGE_PATH --val_annotations=VAL_ANNOTATION_PATH)]
                [--epochs=NUM_EPOCHS --batch_size=BATCH_SIZE --steps_per_epoch=STEPS_PER_EPOCH]
                [--optimizer_name=OPTIMIZER_NAME]
                [--new_checkpoints_dir=CHECKPOINTS_DIR --load_checkpoint=CHECKPOINT_PATH --no_resume_training]
                [(--input_height=INPUT_HEIGHT --input_width=INPUT_WIDTH)]
    keras_segmentation KERASMODEL predict --n_classes=NUM_CLASSES --input_directory=IMAGES_DIR_PATH
                [--output_directory=SEGMENTED_IMAGES_DESTINATION --load_checkpoint=CHECKPOINT_PATH]
                [(--input_height=INPUT_HEIGHT --input_width=INPUT_WIDTH)]

Options:
    -h --help                                         Show this help message.
    --version                                         Display program version.
    --n_classes=NUM_CLASSES                           Number of classes to use for segmentation.
    --train_images=TRAIN_IMAGE_PATH                   Path to images to use for training.
    --train_annotations=TRAIN_ANNOTATION_PATH         Path to image annotations to use for training.
    --validate                                        Whether or to validate performance after each training epoch.
    --val_images=VAL_IMAGE_PATH                       Path to images to use for validation.
    --val_annotations=VAL_ANNOTATION_PATH             Path to image annotations to use for validation.
    --epochs=NUM_EPOCHS                               Number of epochs to train for.
    --batch_size=BATCH_SIZE                           Number of concurrent images to train on.
    --steps_per_epoch=STEPS_PER_EPOCH                 Number of images to train on during an epoch.
    --optimizer_name=OPTIMIZER_NAME                   Optimizer to use during training. Refer to
                                                      https://keras.io/optimizers/
    --new_checkpoints_dir=CHECKPOINTS_DIR             Directory to save new checkpoints during training.
    --no_resume_training                              If --load_checkpoint specified and this argument is passed,
                                                      new checkpoints will start saving with epoch=0. This may
                                                      overwrite existing checkpoints.
    --input_directory=IMAGES_DIR_PATH                 Path to directory containing images to be segmented.
    --output_directory=SEGMENTED_IMAGES_DESTINATION   Path to directory where segmented images are saved.
    --load_checkpoint=CHECKPOINT_PATH                 Path to checkpoint file to load before training or prediction.
    --input_height=INPUT_HEIGHT                       Define specific input height for model. Note, must use same input
                                                      height across training and inference for the same checkpoint.
    --input_width=INPUT_WIDTH                         Define specific input width for model. Note, must use same input
                                                      width across training and inference for the same checkpoint.
"""
from docopt import docopt
from .model_wrapper import WrappedModel

INT_ARGS = ["batch_size", "epochs", "steps_per_epoch"]
NOT_FUNC_PARAMS = ["KERASMODEL", "predict", "train", "--input_height",
                   "--input_width", "--n_classes", "--load_checkpoint"]


if __name__ == "__main__":
    args = docopt(__doc__, version="1.0.0")

    model_args = {
        "keras_model": args['KERASMODEL'],
        "n_classes": int(args['--n_classes'])
    }

    if args['--input_height'] is not None and args['--input_width'] is not None:
        model_args['input_height'] = int(args['--input_height'])
        model_args['input_width'] = int(args['--input_width'])

    model = WrappedModel(**model_args)

    if args['--load_checkpoint'] is not None:
        model.load_weights(args['--load_checkpoint'])

    func_args = {}

    for key, value in args.items():
        if key not in NOT_FUNC_PARAMS and value is not None:
            key = key[2:]

            if key == "no_resume_training":
                func_args['resume_training'] = not value
            elif key == "new_checkpoints_dir":
                func_args['checkpoints_path'] = value
            elif key in INT_ARGS:
                func_args[key] = int(value)
            else:
                func_args[key] = value

    print(func_args)
    if args['predict']:
        func_args = {key: value for key, value in func_args.items() if key not in ['validate', 'resume_training']}
        model.predict_multiple(**func_args)
    elif args['train']:
        model.train_model(**func_args)

    # from . import cli_interface
    # cli_interface.main()
