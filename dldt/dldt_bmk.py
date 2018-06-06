###########################################################################
# This is a sample script file to run Caffe/TF inference benchmarks with 
# mkldnn library on Intel CPUs using DLDT tool
# Please contact vinod.devarampati@intel.com for any clarifications
# Instructions to fill the variable are given in comments
############################################################################
# Mandatory : Install latest DLDT from 
#        - https://software.intel.com/en-us/openvino-toolkit/choose-download
############################################################################

import argparse
import os

models_tf =  {
   "vgg_16":"frozen_vgg_16.xml",
   "vgg_19":"frozen_vgg_19.xml",
   "inception_v3":"frozen_inception_v3.xml",
   "inception_v4":"frozen_inception_v4.xml",
   "resnet_v1_50":"frozen_resnet_v1_50.xml",
   "resnet_v1_101":"frozen_resnet_v1_101.xml",
   "resnet_v1_152":"frozen_resnet_v1_152.xml"
    }

models_cf =  {
    "vgg_16":"VGG_ILSVRC_16_layers.xml",
    "vgg_19":"VGG_ILSVRC_19_layers.xml",
    "inception_v3":"inception-v3.xml",
    "inception_v4":"inception-v4.xml",
    "resnet_v1_50":"ResNet-50-model.xml",
    "resnet_v1_101":"ResNet-101-model.xml",
    "resnet_v1_152":"ResNet-152-model.xml"
    }


def print_results(args):
  """
  """
  files = os.listdir(args.log_dir)

  topology = {}

  for file_name in files:
    if file_name.endswith(".log"):
      with open(os.path.join(args.log_dir,file_name)) as fp:
        data = fp.read()
        start = data.find("iteration:")+11
        if start <= 11:
          continue
        end = data.find(" ",start)
        val = float(data[start:end])

      parse = file_name.split(".")[0].split("_")
      if parse[0] == "resnet":
        top = parse[0]+"_"+parse[1]+"_"+parse[2]
        bs = parse[4]
        stream = parse[5]
      elif parse[0] == "vgg" or parse[0] == "inception":
        top = parse[0]+"_"+parse[1]
        bs = parse[3]
        stream = parse[4]

      if topology.get(top) != None:
        if topology[top].get(bs) != None:
          if topology[top][bs].get(stream) != None:
            topology[top][bs][stream].append(val)
          else: 
            topology[top][bs][stream] = []
            topology[top][bs][stream].append(val)
        else:
          topology[top][bs] = {}
          topology[top][bs][stream] = []
          topology[top][bs][stream].append(val)
      else:
        topology[top] = {}
        topology[top][bs] = {}
        topology[top][bs][stream] = []
        topology[top][bs][stream].append(val)

  for top in topology:
    for bs in topology[top]:
      for stream in topology[top][bs]:
        average_latency = sum(topology[top][bs][stream])/float(len(topology[top][bs][stream]))
        num_streams = float(stream[3:])
        fps = 1000*num_streams//average_latency
        print(top,"\t\t",bs[2:],"\t",num_streams,"\t",average_latency,"\t",fps)


def create_shell_script(args):
  """
  """

  if args.cpu == "skl8180":
    NUM_CORES = 56
    NUM_STREAMS = [1,2,4,8,14,28,56]
  elif args.cpu == "skl6148":
    NUM_CORES = 40
    NUM_STREAMS = [1,2,4,5,8,10,20,40]

  if args.fw == "caffe":
    print("export WKDIR=~/cf_inference_demo")
    model_dict = models_cf
  else: 
    model_dict = models_tf
    print("export WKDIR=~/tf_inference_demo")

  print("export DATA_PATH=~/imageNet")
  print("export SAMPLES_PATH=$WKDIR/samples")
  print("export LOGS_PATH=$WKDIR/logs")
  print("export DLDT_PATH=~/intel/computer_vision_sdk_2018.1.265/deployment_tools")
  print("export MO_MODELS_PATH=$WKDIR/mo_models")
  print("export FROZEN_MODELS=$WKDIR/frozen")
  print("export CAFFE_MODELS=$WKDIR/models")
  print("source  $DLDT_PATH/../bin/setupvars.sh")
  print("source $DLDT_PATH/model_optimizer/install_prerequisites/../venv/bin/activate")


  bs = args.batch_size
  if args.topology != "all":
    model = { args.topology : model_dict[args.topology] }
  else:
    model = model_dict

  for topology in model:  
    for ns in NUM_STREAMS:
      cores_per_stream = NUM_CORES//ns
      print("export OMP_NUM_THREADS="+str(cores_per_stream))
      for i in range(ns):
        j = i*cores_per_stream
        k = (i+1)*cores_per_stream-1
        
        print('export KMP_AFFINITY="granularity=core,proclist=['+str(j)+"-"+str(k)+\
             '],explicit,verbose";$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/'+\
              str(bs)+' -m $MO_MODELS_PATH/'+model[topology]+' -d CPU -nt 2 -ni 100  &>'+\
              topology+"_"+str(i)+"_bs"+str(bs)+"_str"+str(ns)+".log &")
      print("echo 'Waiting for "+str(ns)+"-streams to finish'")
      print("sleep 12s")
      print("ps -elf | grep  samples | for i in $(awk '{print $4}');do kill -9 $i; done")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--cpu", default="skl8180", help="SKU name")
  parser.add_argument("--topology", default="resnet_v1_50", help=" topology name")
  parser.add_argument("--fw", default="caffe", help="caffe/tf")
  parser.add_argument("--batch_size", type=int, default=1, help="i Batch size")
  parser.add_argument("--mode", type=str, default="exe", help="exe/log")
  parser.add_argument("--log_dir", type=str, default="./", help="logs directory")
  args = parser.parse_args()
  if args.mode == "exe":
    create_shell_script(args)
  else:
    print_results(args)
