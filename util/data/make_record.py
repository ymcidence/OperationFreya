from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import tarfile
import tempfile
from urllib.request import urlretrieve
import os

from absl import app
from absl import flags
import numpy as np
import scipy.io
import tensorflow as tf
from util.data import processing
from util.data import path

flags.DEFINE_string(
    "directory",
    path.BUILD_TFRECORDS_DOWNLOAD_PATH,
    "Directory to download and write to.",
)

flags.DEFINE_integer("seed", 0, "Random seed for determinism.")

flags.DEFINE_string("dataset_name", "cifar10", "Name of dataset")

FLAGS = flags.FLAGS

COUNTS = {
    "svhn": {"train": 73257, "test": 26032, "valid": 7326, "extra": 531131},
    "cifar10": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0},
    "imagenet_32": {
        "train": 1281167,
        "test": 50000,
        "valid": 50050,
        "extra": 0,
    },
    "cifar_unnormalized": {
        "train": 50000,
        "test": 10000,
        "valid": 5000,
        "extra": 0,
    },
    "mnist": {
        "train": 60000,
        "test": 10000,
        "valid": 5000,
        "extra": 0,
    },
}

URLS = {
    "svhn": "http://ufldl.stanford.edu/housenumbers/{}_32x32.mat",
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz",
}

_DATA_DIR = "data/imagenet_32"


def _load_imagenet_32():
    train_file_names = ["train_data_batch_" + str(idx) for idx in range(1, 11)]
    all_image_data_val = np.load(
        os.path.join(_DATA_DIR, "val_data" + "_image.npy")
    )
    all_label_data_val = np.load(
        os.path.join(_DATA_DIR, "val_data" + "_label.npy")
    )
    image_data_list_train = []
    label_data_list_train = []
    for file_name in train_file_names:
        image_data_list_train.append(
            np.load(os.path.join(_DATA_DIR, file_name + "_image.npy"))
        )
        label_data_list_train.append(
            np.load(os.path.join(_DATA_DIR, file_name + "_label.npy"))
        )
    all_image_data_train = np.concatenate(image_data_list_train)
    all_label_data_train = np.concatenate(label_data_list_train)
    all_image_data_train = np.transpose(
        all_image_data_train.reshape((-1, 3, 32, 32)), [0, 2, 3, 1]
    )
    all_image_data_val = np.transpose(
        all_image_data_val.reshape((-1, 3, 32, 32)), [0, 2, 3, 1]
    )
    train_set = {
        "images": np.reshape(all_image_data_train, (-1, 32, 32, 3)),
        "labels": all_label_data_train - 1,
    }
    test_set = {
        "images": np.reshape(all_image_data_val, (-1, 32, 32, 3)),
        "labels": all_label_data_val - 1,
    }
    return train_set, test_set


def _load_mnist():
    data = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data

    train_set = {
        "images": x_train[..., np.newaxis],
        "labels": y_train,
    }

    test_set = {
        "images": x_test[..., np.newaxis],
        "labels": y_test,
    }

    return train_set, test_set


def _load_svhn():
    splits = collections.OrderedDict()
    for split in ["train", "test", "extra"]:
        with tempfile.NamedTemporaryFile() as f:
            urlretrieve(URLS["svhn"].format(split), f.name)
            data_dict = scipy.io.loadmat(f.name)
        dataset = {}
        dataset["images"] = np.transpose(data_dict["X"], [3, 0, 1, 2])
        dataset["labels"] = data_dict["y"].reshape((-1))
        # SVHN raw data uses labels from 1 to 10; use 0 to 9 instead.
        dataset["labels"][dataset["labels"] == 10] = 0
        splits[split] = dataset
    return splits.values()


def _load_cifar10(normalize):
    def unflatten(images):
        return images.reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        urlretrieve(URLS["cifar10"], f.name)
        tar = tarfile.open(fileobj=f)
        train_data_batches, train_data_labels = [], []
        for batch in range(1, 6):
            data_dict = scipy.io.loadmat(
                tar.extractfile(
                    "cifar-10-batches-mat/data_batch_{}.mat".format(batch)
                )
            )
            train_data_batches.append(data_dict["data"])
            train_data_labels.append(data_dict["labels"].flatten())
        train_set = {
            "images": np.concatenate(train_data_batches, axis=0),
            "labels": np.concatenate(train_data_labels, axis=0),
        }
        data_dict = scipy.io.loadmat(
            tar.extractfile("cifar-10-batches-mat/test_batch.mat")
        )
        test_set = {
            "images": data_dict["data"],
            "labels": data_dict["labels"].flatten(),
        }
    if normalize:
        train_set["images"] = processing.gcn(train_set["images"])
        test_set["images"] = processing.gcn(test_set["images"])
        zca_transform = processing.get_zca_transformer(
            train_set["images"], root_path=FLAGS.directory
        )
        train_set["images"] = zca_transform(train_set["images"])
        test_set["images"] = zca_transform(test_set["images"])
    train_set["images"] = unflatten(train_set["images"])
    test_set["images"] = unflatten(test_set["images"])
    train_set["images"] = train_set["images"].astype(
        processing.DATASET_DTYPE["cifar10"].as_numpy_dtype
    )
    test_set["images"] = test_set["images"].astype(
        processing.DATASET_DTYPE["cifar10"].as_numpy_dtype
    )
    return train_set, test_set


def main(_):
    rng = np.random.RandomState(FLAGS.seed)

    train_count = COUNTS[FLAGS.dataset_name]["train"]
    validation_count = COUNTS[FLAGS.dataset_name]["valid"]
    test_count = COUNTS[FLAGS.dataset_name]["test"]
    extra_count = COUNTS[FLAGS.dataset_name]["extra"]

    extra_set = None  # In general, there won't be extra data.
    if FLAGS.dataset_name == "svhn":
        train_set, test_set, extra_set = _load_svhn()
    elif FLAGS.dataset_name == "cifar10":
        train_set, test_set = _load_cifar10(normalize=True)
    elif FLAGS.dataset_name == "cifar_unnormalized":
        train_set, test_set = _load_cifar10(normalize=False)
    elif FLAGS.dataset_name == "imagenet_32":
        train_set, test_set = _load_imagenet_32()
    elif FLAGS.dataset_name == "mnist":
        train_set, test_set = _load_mnist()
    else:
        raise ValueError("Unknown dataset", FLAGS.dataset_name)

    # Shuffle the training data
    indices = rng.permutation(train_set["images"].shape[0])
    train_set["images"] = train_set["images"][indices]
    train_set["labels"] = train_set["labels"][indices]

    # If the extra set exists, shuffle it.
    if extra_set is not None:
        extra_indices = rng.permutation(extra_set["images"].shape[0])
        extra_set["images"] = extra_set["images"][extra_indices]
        extra_set["labels"] = extra_set["labels"][extra_indices]

    # Split the training data into training and validation data
    train_images = train_set["images"][validation_count:]
    train_labels = train_set["labels"][validation_count:]
    validation_images = train_set["images"][:validation_count]
    validation_labels = train_set["labels"][:validation_count]
    validation_set = {"images": validation_images, "labels": validation_labels}
    train_set = {"images": train_images, "labels": train_labels}

    # Convert to Examples and write the result to TFRecords.
    processing.convert_to(
        train_set["images"],
        train_set["labels"],
        train_count - validation_count,
        "train",
        FLAGS.directory,
        FLAGS.dataset_name,
    )

    processing.convert_to(
        test_set["images"],
        test_set["labels"],
        test_count,
        "test",
        FLAGS.directory,
        FLAGS.dataset_name,
    )

    processing.convert_to(
        validation_set["images"],
        validation_set["labels"],
        validation_count,
        "validation",
        FLAGS.directory,
        FLAGS.dataset_name,
    )

    if extra_set is not None:
        processing.convert_to(
            extra_set["images"],
            extra_set["labels"],
            extra_count,
            "extra",
            FLAGS.directory,
            FLAGS.dataset_name,
        )


if __name__ == "__main__":
#     FLAGS.dataset_name = "mnist"
    app.run(main)
