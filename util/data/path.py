from meta import ROOT_PATH
import os

TRAIN_DIR = os.path.join(ROOT_PATH, "data")
IMAGENET_DATA_DIR = os.path.join(TRAIN_DIR, "imagenet-2012-tfrecord")
IMAGENET_32_DATA_DIR = os.path.join(TRAIN_DIR, "imagenet_32")

BUILD_TFRECORDS_DOWNLOAD_PATH = os.path.join(ROOT_PATH, "data/")
BUILD_TFRECORDS_DATA_PREFIX = os.path.join(ROOT_PATH, "data/")
RAW_IMAGENET_PATH = os.path.join(ROOT_PATH, "data/raw/imagenet/")
LABEL_MAP_PATH = BUILD_TFRECORDS_DOWNLOAD_PATH
