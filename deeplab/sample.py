###  COPY ALL THE CODE INTO A JYPYTER NOTEBOOK  ### 
###  THE JYPYTER NOTEBOOK NEEDS TO BE IN 'tensorflow\models\research\deeplab'  ### 

## Imports

import collections
import os
import io
import sys
import tarfile
import tempfile
import urllib

from IPython import display
from ipywidgets import interact
from ipywidgets import interactive
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
# import skvideo.io

import tensorflow as tf

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

# Needed to show segmentation colormap labels
sys.path.append('utils')
import get_dataset_colormap


## Select and download models


# _MODEL_URLS = {
#     'xception_coco_voctrainaug': 'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
#     'xception_coco_voctrainval': 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz',
# }
# 
# Config = collections.namedtuple('Config', 'model_url, model_dir')
# 
# def get_config(model_name, model_dir):
#     return Config(_MODEL_URLS[model_name], model_dir)
# 
# config_widget = interactive(get_config, model_name=_MODEL_URLS.keys(), model_dir='')
# display.display(config_widget)
# 
# # Check configuration and download the model
# 
# _TARBALL_NAME = 'deeplab_model.tar.gz'
# 
# config = config_widget.result
# 
# model_dir = config.model_dir or tempfile.mkdtemp()
# tf.gfile.MakeDirs(model_dir)
# 
# download_path = os.path.join(model_dir, _TARBALL_NAME)
# print('downloading model to %s, this might take a while...' % download_path)
# urllib.request.urlretrieve(config.model_url, download_path)
# print('download completed!')

MODEL_NAME = 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
download_path = os.path.join(sys.path[0], MODEL_NAME)

## Load model in TensorFlow

_FROZEN_GRAPH_NAME = 'frozen_inference_graph'


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        config = tf.ConfigProto(
                gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.8,
                    visible_device_list="0",
                    allow_growth=True
            )
        )

        self.graph = tf.Graph()
        
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if _FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()
        
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():      
            tf.import_graph_def(graph_def, name='')
        
        self.sess = tf.Session(graph=self.graph, config=config)
            
    def run(self, image):
        """Runs inference on a single image.
        
        Args:
            image: A PIL.Image object, raw input image.
            
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

model = DeepLabModel(download_path)


## Webcam demo

cap = cv2.VideoCapture(0)

# Next line may need adjusting depending on webcam resolution
final = np.zeros((1, 384, 1026, 3))
while True:
    ret, frame = cap.read()
    
    # From cv2 to PIL
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    
    # Run model
    resized_im, seg_map = model.run(pil_im)
    
    # Adjust color of mask
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    
    # Convert PIL image back to cv2 and resize
    frame = np.array(pil_im)
    r = seg_image.shape[1] / frame.shape[1]
    dim = (int(frame.shape[0] * r), seg_image.shape[1])[::-1]
    # resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    resized = cv2.resize(frame, dim)
    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    
    # Stack horizontally color frame and mask
    color_and_mask = np.hstack((resized, seg_image))

    cv2.imshow('frame', color_and_mask)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
