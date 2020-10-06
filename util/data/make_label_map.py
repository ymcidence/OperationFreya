from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
import tensorflow as tf
from absl import logging
import numpy as np

from util.data import processing as dataset_utils
from util.data import path as paths

flags.DEFINE_string("dataset_name", "default", "Name of source dataset.")
flags.DEFINE_string(
    "n_labeled_list",
    "20,50,100,250,500,1000,2000,4000,8000",
    "Comma-separated list of label counts to create label maps for."
)
flags.DEFINE_integer("label_map_index", 0, "Identifier for this label map.")
flags.DEFINE_integer("seed", 0, "Random seed for determinism.")
flags.DEFINE_string(
    "fkeys_path",
    paths.LABEL_MAP_PATH,
    "Where to write read the fkeys and write the label_maps.",
)
flags.DEFINE_string(
    "imagenet_path",
    paths.RAW_IMAGENET_PATH,
    "Where to read raw imagenet files.",
)

FLAGS = flags.FLAGS


def main(_):
    rng = np.random.RandomState(FLAGS.seed)
    # Build a label map for each label_count, with several seeds
    for n_labeled in [int(n) for n in FLAGS.n_labeled_list.split(',')]:
        build_single_label_map(
            n_labeled,
            FLAGS.label_map_index,
            FLAGS.dataset_name,
            FLAGS.imagenet_path,
            FLAGS.fkeys_path,
            rng,
        )


def build_single_label_map(
        n_labeled, label_map_index, dataset_name, imagenet_path, fkeys_path, rng
):
    """Builds just one label map - we call this in a larger loop.
    As a side effect, this function writes the label map to a file.
    Args:
        n_labeled: An integer representing the total number of labeled
            examples desired.
        label_map_index: An integer representing the index of the label map.
            We may want many label_maps w/ same value of n_labeled.
            This allows us to disambiguate.
        dataset_name: A string representing the name of the dataset.
            One of 'cifar10', 'svhn', 'imagenet', or 'cifar_unnormalized'.
        imagenet_path: A string that encodes the location of the raw imagenet
            data.
        fkeys_path: A string that encodes where to read fkeys from and write
            label_maps to.
        rng: np.random.RandomState instance for drawing random numbers.
    Raises:
        ValueError: if passed an unrecognized dataset_name.
    """
    # Set the name of the label_map
    # This will be named as label_map_count_{count}_idx_{idx}
    destination_name = "label_map_count_{}_index_{}".format(
        n_labeled, label_map_index
    )
    result_dict = {"values": []}
    n_labeled_per_class = (
            n_labeled // dataset_utils.DATASET_CLASS_COUNT[dataset_name]
    )

    if dataset_name == "imagenet":
        synsets = tf.io.gfile.listdir(imagenet_path)
        for synset in synsets:
            logging.info("processing: %s", synset)
            unique_ids = [
                f[: -len(".JPEG")]
                for f in tf.io.gfile.listdir(os.path.join(imagenet_path, synset))
            ]
            random_ids = rng.choice(
                len(unique_ids), n_labeled_per_class, replace=False
            )
            result_dict["values"] += [unique_ids[n] for n in random_ids]
    elif dataset_name in {"cifar10", "svhn", "cifar_unnormalized", "mnist"}:
        path = os.path.join(fkeys_path, dataset_name, "label_to_fkeys_train")
        with gfile.GFile(path, "r") as f:
            label_to_fkeys = json.load(f)
        for label in label_to_fkeys.keys():
            random_ids = rng.choice(
                len(label_to_fkeys[label]), n_labeled_per_class, replace=False
            )
            result_dict["values"] += [
                label_to_fkeys[label][n] for n in random_ids
            ]
    else:
        raise ValueError("Dataset not supported: {}.".format(dataset_name))

    # Save the results in a JSON file
    result_path = os.path.join(fkeys_path, dataset_name, destination_name)
    with gfile.GFile(result_path, "w") as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
#     FLAGS.dataset_name = "mnist"
    for i in range(10):
        FLAGS.label_map_index = i
        FLAGS.seed = i
        main(0)
