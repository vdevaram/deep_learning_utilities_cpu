# This file consists of algorithm to run Face recognition neural network
# modified by vinod.devarampati@intel.com for flexible batch sizes
from __future__ import print_function
import numpy as np
import caffe
import argparse
import math
import os.path
import time

class feature_extraction:
  """
  """
  def __init__(self,
               image_list_file,
               graph,
               weights,
               num_channels,
               height,
               width,
               mean,
               scale,
               batch_size,
               features_file,
               hw):
    """
    """
    self._batch_size = batch_size
    self._proto = graph
    self._params = weights
    self._num_channels = num_channels
    self._img_height = height
    self._img_width = width
    self._scale = scale
    self._mean = np.array([mean, mean, mean])
    self._features = self.load_features(features_file)
    self._features_file = features_file
    self._image_list = self.load_images(image_list_file)
    self._net = None
    self._transformer = None
    self._index = 0
    self._update = True

    self.set_mode(hw)
    
  def set_mode(self,hw):
    """
    """
    if hw == 'CPU':
      caffe.set_mode_cpu()
    elif hw == 'GPU':
      caffe.set_mode_gpu()

  def load_images(self,image_list_file):
    """
    """
    with open(image_list_file) as fp:
      return (fp.readlines())

  def load_features(self,features_file):
    """
    """
    features = []
    if os.path.exists(features_file):
      with open(features_file,"r") as fp:
        lines = fp.readlines()

      for line in lines:
        name,norm,feature_vec = line.split('@')
        norm = float(norm)
        temp = feature_vec.rstrip().split(" ")
        temp  = np.array(temp)
        feature_vec = temp.astype(float)
        feature_dict = {"name":name, "norm":norm, "feature_vec":feature_vec}
        features.append(feature_dict)

    return features

  def init_net(self):
    self._net = caffe.Net(self._proto, self._params, caffe.TEST)
    self._net.blobs['data'].reshape(self._batch_size, self._num_channels, self._img_height, self._img_width)
    self._transformer =  caffe.io.Transformer({'data': self._net.blobs['data'].data.shape})
    self._transformer.set_transpose('data',(2,0,1))
    self._transformer.set_mean('data', self._mean)
    self._transformer.set_raw_scale('data', 255)
    self._transformer.set_channel_swap('data', (2,1,0))
    
  def detect_face(self):
    """
    """
    self.init_net()
    #start = time.time()
    for i in range(len(self._image_list)//self._batch_size):
      batch_time = time.time()
      self.detect_batch()
      print("FPS for batch-",i,": ",self._batch_size//(time.time()-batch_time))



  def detect_batch(self):  
    """
    """
    self.load_batch() 
    output = self._net.forward()
    for i in range(self._batch_size):
      image_features = output['fc5'][i].copy()
      norm = math.sqrt(np.dot(image_features,image_features))
      identity = self.search_face(image_features,norm)
      if identity == None:
        name = self._image_list[self._index+i].rstrip()
        feature_dict = {"name":name, "norm":norm, "feature_vec":image_features}
        self._features.append(feature_dict)
        if self._update:
          with open(self._features_file,"a+") as fp:
            fp.write(name+"@"+str(round(norm,4))+"@")
            for val in image_features:
              fp.write(str(round(val,4))+" ")
            fp.write("\n")
      else:
        with open("result.txt","a+") as fp:
          #fp.write(os.path.split(identity)[1]+"\n")
          fp.write(identity+"\n")
    self._index = self._index + self._batch_size

  def load_batch(self):
    """
    """
    transformed_image = []
    for i in range(self._batch_size):
      image = caffe.io.load_image(self._image_list[self._index+i].rstrip())
      transformed_image.append(self._transformer.preprocess('data', image) * self._scale)

    self._net.blobs['data'].data[...] = transformed_image
   
  def search_face(self,image_features,norm):
    """
    """
    ret = None
    for face in self._features:
      corr = np.dot(face['feature_vec'],image_features)
      cos_dist = corr/(norm * face['norm'])
      if cos_dist > 0.6 :
        ret = face['name']
        break;

    return ret
      
     


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--image_list_file", default="lfw_image_list.txt",help="file path consists of list of images to be processsed")
  parser.add_argument("--graph", default="kscf.prototxt",help="graph/model to be executed")
  parser.add_argument("--weight",default="kscf.caffemodel",help="trained_weights")
  parser.add_argument("--input_channels", type=int,default=3, help="input height")
  parser.add_argument("--input_height", type=int,default=112, help="input height")
  parser.add_argument("--input_width", type=int, default=96,help="input width")
  parser.add_argument("--input_mean", type=float,default=127.5, help="input mean")
  parser.add_argument("--input_scale", type=float,default=0.0078125, help="input scale")
  parser.add_argument("--batch_size", type=int, default=1,help="batch_size")
  parser.add_argument("--features_file", default="face_records.rec", help="File path to store Facial features")
  parser.add_argument("--hw", default="CPU", help="Hardware to use CPU/GPU")
  args = parser.parse_args()
  
  features = feature_extraction(
                       args.image_list_file,
                       args.graph,
                       args.weight,
                       args.input_channels,
                       args.input_height,
                       args.input_width,
                       args.input_mean,
                       args.input_scale,
                       args.batch_size,
                       args.features_file,
                       args.hw)

  features.detect_face()
