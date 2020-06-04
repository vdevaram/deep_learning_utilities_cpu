# TF code to run inference with saved_model dir
# This code uses the resent50 saved model dir from http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW_jpg.tar.gz

# ######## instructions to run ##########
# export WORK_DIR=<some path> ex: export WORK_DIR=$(pwd)
# cd $WORK_DIR
# wget http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW_jpg.tar.gz
# tar -zxvf resnet_v2_fp32_savedmodel_NCHW_jpg.tar.gz
# export OMP_NUM_THREADS=<num threads to run inference on> ex: export OMP_NUM_THREADS=20
# export KMP_BLOCKTIME=1
# export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
# numactl -m 0 -N 0 -C 0-19 python resnet_local.py --image_path <image path> --model_dir $WORK_DIR/resnet_v2_fp32_savedmodel_NCHW_jpg/1538687370/ --input_layer map/Shape --output_layer ArgMax --inter_threads 1 --intra_threads $OMP_NUM_THREADS
# ######## end #########

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import time
import os


def run_inference(args):
  " "
  with open(args.image_path, 'rb') as f:
    data = f.read()
  
  config = tf.ConfigProto()
  config.intra_op_parallelism_threads = args.intra_threads
  config.inter_op_parallelism_threads = args.inter_threads
  config.allow_soft_placement = True
  config.experimental.use_numa_affinity = True

  with tf.Session(config = config, graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], args.model_dir)
    graph = tf.get_default_graph()
    in_op = graph.get_operation_by_name(args.input_layer)
    out_op = graph.get_operation_by_name(args.output_layer)
    result = sess.run(out_op.outputs[0], {in_op.inputs[0] : [data]})
    start = time.time()
    for i in range(100):
      #s = time.time()
      result = sess.run(out_op.outputs[0], {in_op.inputs[0] : [data]})
      #print(time.time() - s)
    dur = (time.time() - start)/100
    print ("duration = ", dur," FPS: ", 1/dur)    
    print("label", result)
 

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_path", help="image DIR to be processed")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  parser.add_argument("--model_dir", help="path of saved_model dir")
  parser.add_argument("--inter_threads", type=int, help="inter threads")
  parser.add_argument("--intra_threads", type=int, help="intra threads")
  args = parser.parse_args()

  run_inference(args)
