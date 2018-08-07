#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys, os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

import cv2
import rospy
from sensor_msgs.msg import Image as rosImage
from cv_bridge import CvBridge, CvBridgeError

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.config = tf.ConfigProto(
                gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.4,
                    visible_device_list="0",
                    allow_growth=True
            )
        )

    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(config=self.config, graph=self.graph)

  def run(self, image):
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

def create_pascal_label_colormap():
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)
  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  
  cv_image      = np.asarray(image)
  cv_seg_image  = np.asarray(seg_image)
  cv_mask_image = cv2.addWeighted(cv_seg_image, 0.9, cv_image, 0.8, 0)

  return cv_mask_image
  
  """
  {{{
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()
  }}}
  """
class Segmentation(object):
    def __init__(self):
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", rosImage, self.imageCallback, queue_size=10)
        self.image_pub = rospy.Publisher("/deeplab/image", rosImage, queue_size=10)

    def imageCallback(self, image_msg):
        try:
            self.cv_Image = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")
            self.pil_Image = Image.fromarray(cv2.cvtColor(self.cv_Image, cv2.COLOR_BGR2RGB))
            resized_im, seg_map = MODEL.run(self.pil_Image)
            seg_image = vis_segmentation(resized_im, seg_map)
            
            pub_image = CvBridge().cv2_to_imgmsg(seg_image, "bgr8")
            self.image_pub.publish(pub_image)

        except CvBridgeError as e:
            print (e)

    def main(self):
        rospy.init_node("deeplab_ros")
        rospy.spin()

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

MODEL_NAME = 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
# MODEL_NAME = 'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz'
download_path = os.path.join(sys.path[0], MODEL_NAME)
MODEL = DeepLabModel(download_path)

def main():
    segmentation = Segmentation()
    segmentation.main()

if __name__ == '__main__':
    print("start")
    main()
