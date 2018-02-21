# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# modified by vinod.devarampati@intel.com for flexible batch sizes

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import time
import os


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_list,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255,
                                batch_size=1,
                                index=0):

 
  im = []
  print ("Batch start index :",index)
  for i in range(batch_size):

    file_name = file_list[i+index]
    
    file_reader = tf.read_file(file_name)
    if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(
          file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(
          tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
      image_reader = tf.image.decode_jpeg(
          file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    resized = tf.image.resize_images(float_caster, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    im.append(normalized)
  sess = tf.Session()
  result = sess.run(im)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":

 
  
  model_file = "frozen/frozen_resenet_v1_101.pb"
  label_file = "imagenet_slim_labels.txt"
  output_layer = "resnet_v1_101/predictions/Reshape_1"
  # assuming all images are stored in images folder
  image_dir = "images"
  input_height = 224
  input_width = 224
  input_mean = 0
  input_std = 255
  input_layer = "input"  
  batch_size = 1
  top_accuracy = 1


  parser = argparse.ArgumentParser()
  parser.add_argument("--image_dir", help="image DIR to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  parser.add_argument("--batch_size", type=int, help="batch_size")
  parser.add_argument("--top_accuracy", type=int, help="number of top accuracy")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image_dir:
    if not os.path.isdir(args.image_dir):
      print (args.image_dir, " is not a directory for getting images list\n")
      exit()
    image_dir = args.image_dir
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer
  if args.batch_size:
    batch_size = args.batch_size
  if args.top_accuracy:
    top_accuracy = args.top_accuracy
  top_accuracy = -1 * top_accuracy  

  graph = load_graph(model_file)

  file_list = os.listdir(image_dir)
  file_list = [ os.path.join(image_dir,i) for i in file_list ]

  all_start = time.time()
  
  length = len(file_list) // batch_size

  for i in range(length):
    
    start = time.time()
    i = i * batch_size
    t = read_tensor_from_image_file(
      file_list,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std,
      batch_size = batch_size,
      index = i)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
      })
    labels = load_labels(label_file)
    for j in range(batch_size):
      top_k = results[j].argsort()[top_accuracy:][::-1]
      for k in top_k:
        print(labels[k], results[j][k])
    print("time for batch is ",time.time() - start,"ms\n\n")
  print("time for all images is ",time.time() - all_start,"ms\n\n")
