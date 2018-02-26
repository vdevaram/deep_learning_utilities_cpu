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
                                index=0,
                                image_type="jpeg"):

 
  image = tf.placeholder(tf.string)
  if image_type == "jpeg":
    decode = tf.image.decode_jpeg(image,channels=3)
  elif image_type == "png":
    decode = tf.image.decode_png(image,channels=3)
  elif image_type == "bmp":
    decode = tf.image.decode_bmp(image,channels=3)
  else :
    print (image_type, " is not handled")
    exit()
  resize = tf.image.resize_images(decode, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resize, [input_mean]), [input_std])
  with tf.Session() as sess:
    result =[]
    for num in range(batch_size):
      img_data = tf.gfile.FastGFile(file_list[num+index],"rb").read()
      out = sess.run(normalized,feed_dict = {image:img_data})
      result.append(out)

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
  image_type = "jpeg"
  input_height = 224
  input_width = 224
  input_mean = 0
  input_std = 255
  input_layer = "input"  
  batch_size = 1
  top_accuracy = 1
  data_type = "ImageNet"
  benchmark = 1


  parser = argparse.ArgumentParser()
  parser.add_argument("--image_dir", help="image DIR to be processed")
  parser.add_argument("--image_type", help="images type to be processed. Supported are jpeg/png/bmp")
  parser.add_argument("--data_type", help="input data type to be processed 'synthetic or imagenet'")
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
  parser.add_argument("--benchmark", type=int, help=" 0 if inference results to be printed")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image_dir:
    if not os.path.isdir(args.image_dir):
      print (args.image_dir, " is not a directory for getting images list\n")
      exit()
    image_dir = args.image_dir
  if args.image_type:
    image_type = args.image_type
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
  if args.data_type:
    data_type = args.data_type
  if not args.benchmark:
    benchmark = args.benchmark
  if args.top_accuracy:
    top_accuracy = args.top_accuracy
  top_accuracy = -1 * top_accuracy  

  graph = load_graph(model_file)

  file_list = os.listdir(image_dir)
  file_list = [ os.path.join(image_dir,i) for i in file_list ]

  start = time.time()
  
  length = len(file_list) // batch_size
  for i in range(length):
    
    print ("Batch :",i)
    batch_start = time.time()
    i = i * batch_size
    if data_type == "synthetic":
      data = tf.truncated_normal([batch_size, input_height, input_width, 3],
                                 dtype = tf.float32,
                                 mean = input_mean,
                                 stddev = input_std)
      with tf.Session() as sess:
        t = sess.run (data)

    else :
      t = read_tensor_from_image_file(
      file_list,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std,
      batch_size = batch_size,
      index = i,
      image_type=image_type)
    pre_time = time.time()
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
      })
    inf_time = time.time()
    if benchmark == 0:
      labels = load_labels(label_file)
      for j in range(batch_size):
        top_k = results[j].argsort()[top_accuracy:][::-1]
        for k in top_k:
          print(labels[k], results[j][k])
    batch_time = time.time()
    print("Data TP\t\t","Inference TP\t\t","Batch TP\t")
    print(batch_size//(pre_time-batch_start),"\t\t",batch_size//(inf_time-pre_time),"\t\t\t",batch_size//(batch_time-batch_start),"\t\n\n")
  print("Total Throughput of ",length*batch_size," "+ data_type + " data: ", length*batch_size/(time.time()-start)," Images/sec")  
