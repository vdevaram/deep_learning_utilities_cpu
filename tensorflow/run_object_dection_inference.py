import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import argparse

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util

from utils import visualization_utils as vis_util

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS =  '/home/vinod/tf/models/research/object_detection/data/mscoco_label_map.pbtxt'
PATH_TO_TEST_IMAGES_LIST_FILE = "/home/vinod/tf/image_list.txt"
TOPOLOGY = \
{\
    "FRCNN_RES50" : 
        "/home/vinod/dldt/object_detection/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb",
    "SSD_MOBNET_V2" : 
        "/home/vinod/dldt/object_detection/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
    "RFCN_RES101" : 
        "/home/vinod/dldt/object_detection/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb",
    "MASK_RCNN_RES50" : 
        "/home/vinod/dldt/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28/frozen_inference_graph.pb",
    "MASK_RCNN_RES101": 
        "/home/vinod/dldt/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28/frozen_inference_graph.pb",
}

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph, args):
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = 18
  config.inter_op_parallelism_threads = 2
  config.experimental.use_numa_affinity = True

  with graph.as_default():
    with tf.Session(config=config) as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      if args.benchmark : 
        start = time.time()
        for i in range(100):
          output_dict = sess.run(tensor_dict,
                          feed_dict={image_tensor: np.expand_dims(image, 0)})
        duration = time.time() - start
        print("Time for iteration is ", duration*10," ms and FPS is ", 100/duration," images/sec") 
      else :
        output_dict = sess.run(tensor_dict,
                        feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--image_list", help="txt file with image list to be processed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--topology", help="Topology")
  parser.add_argument("--benchmark", help="benchmark list of topologies")

  args = parser.parse_args()

  if not args.image_list:
      args.image_list = PATH_TO_TEST_IMAGES_LIST_FILE
  if not os.path.isfile(args.image_list) :
      raise ValueError ("file does not exists at ", args.image_list)
  if not args.topology :
      args.topology = "SSD_MOBNET_V2"
  graph = TOPOLOGY[args.topology] 
  if not os.path.isfile(graph) :
      raise ValueError ("file does not exists at ", graph)
  if not args.labels :
      args.labels = PATH_TO_LABELS
  if not os.path.isfile(args.labels) :
      raise ValueError ("file does not exists at ", args.labels)

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(graph, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  category_index = label_map_util.create_category_index_from_labelmap(args.labels, use_display_name=True)

  with open(args.image_list) as fp:
      image_paths = fp.readlines()

  for path in image_paths:
    path = path.strip('\n')
    print("processing image : ", path)
    image = Image.open(path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph, args)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
    #plt.figure(figsize=IMAGE_SIZE)
    plt.imsave(path.split('.')[0]+"_"+args.topology+".png",image_np)
    if args.benchmark :
        break;
