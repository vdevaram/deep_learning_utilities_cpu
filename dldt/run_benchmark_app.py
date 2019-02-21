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
   "resnet_v1_152":"frozen_resnet_v1_152.xml",
   "frcnn_res_50":"frozen_frcnn_res50.xml"
    }

models_tf_custom =  {
   "inception_v3":"frozen_inception_v3.xml",
   "resnet_v1_50":"frozen_resnet_v1_50.xml",
   "inception_resnet_v2":"frozen_inception_resnet_v2.xml",
   "frcnn_res_50":"frozen_frcnn_resnet50.xml",
   "ssd_mobilenet":"frozen_ssd_mobnet.xml",
   "yolo_v2":"frozen_darknet_yolov2_model.xml",
   "yolo_tiny_v2":"frozen_yolov2-tiny-voc.xml",
   "yolo_v3":"frozen_darknet_yolov3_model.xml",
   "yolo_tiny_v3":"frozen_darknet_yolov3_tiny_model.xml",
   "rfcn":"frozen_rfcn_graph.xml",
    }

models_cf =  {
    "vgg_16":"VGG_ILSVRC_16_layers.xml",
    "vgg_19":"VGG_ILSVRC_19_layers.xml",
    "inception_v3":"inception-v3.xml",
    "inception_v4":"inception-v4.xml",
    "resnet_v1_50":"ResNet-50-model.xml",
    "resnet_v1_101":"ResNet-101-model.xml",
    "resnet_v1_152":"ResNet-152-model.xml",
    "ssd_vgg_16": "VGG_VOC0712_SSD_300x300_iter_120000.xml"
    }

models_cf_custom =  {
    "inception_v3":"inception-v3.xml",
    "resnet_v1_50":"ResNet-50-model.xml",
    "ssd_vgg_16": "VGG_VOC0712_SSD_300x300_iter_120000.xml"
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
        start = data.find("Throughput:")+12
        if start <= 12:
          continue
        end = data.find(" ",start)
        val = float(data[start:end])

      parse = file_name.split(".")[0].split("_")
      if parse[0] == "resnet" or parse[0] == "frcnn" or parse[1] == "resnet" or parse[1] == "tiny":
        top = parse[0]+"_"+parse[1]+"_"+parse[2]
        bs = parse[3]
        sync_type = parse[4]
        if sync_type != "sync":
          stream = parse[5]
        else:
          stream = "req0"
      elif parse[0] == "vgg" or parse[0] == "inception" or parse[0] == "ssd" or parse[0] == "yolo":
        top = parse[0]+"_"+parse[1]
        bs = parse[2]
        sync_type = parse[3]
        if sync_type != "sync":
          stream = parse[4]
        else:
          stream = "req0"
      elif parse[0] == "rfcn":
        top = parse[0]
        bs = parse[1]
        sync_type = parse[2]
        if sync_type != "sync":
          stream = parse[3]
        else:
          stream = "req0"

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
        num_streams = int(stream[3:])
        if num_streams != 0:
         print(top,"\t\t",bs[2:],"\t",num_streams,"\t",topology[top][bs][stream][0])
        else:
         print(top,"\t\t",bs[2:],"\t","sync","\t",topology[top][bs][stream][0],"\t",1000/topology[top][bs][stream][0])


def create_shell_script(args):
  """
  """

  bs = args.batch_size
  print("export WKDIR=~/dldt")
  if args.fw == "caffe":
    print("export MO_MODELS_PATH=$WKDIR/cf_mo_models")
    if args.topology == "all":
      model = models_cf
    elif args.topology == "custom":
      model = models_cf_custom
    else:
      model = { args.topology : models_cf[args.topology] }
  else:
    print("export MO_MODELS_PATH=$WKDIR/tf_mo_models")
    if args.topology == "all":
      model = models_tf
    elif args.topology == "custom":
      model = models_tf_custom
    else:
      model = { args.topology : models_tf[args.topology] }

  LOGS_PATH="$WKDIR/logs"
  print("export DATA_PATH=$WKDIR/imageNet")
  print("export SAMPLES_PATH=$WKDIR/samples")
  print("export DLDT_PATH=~/intel/computer_vision_sdk/deployment_tools")
  print("source  $DLDT_PATH/../bin/setupvars.sh")


  if args.cpu == "skl6148" or "skl6248":
    NUM_CORES = 40
    if bs == 1:
      NUM_STREAMS = [1,2,4,5,8,10,20,40]
    elif bs == 8:
      NUM_STREAMS = [1,2,4,5]
    elif bs == 20:
      NUM_STREAMS = [1,2,4]
    elif bs == 40:
      NUM_STREAMS = [1,2,4]

  for topology in model:
    if topology == "ssd_mobilenet" or topology == "frcnn_res_50" or topology == "rfcn":
      sleep_time = "21s"
    else:
      sleep_time = "11s"

    executable = "benchmark_app"

    for ns in NUM_STREAMS:
      log_file = os.path.join(LOGS_PATH,topology+"_bs"+str(bs)+"_async_req"+str(ns)+".log &")
      print('$SAMPLES_PATH/intel64/Release/'+executable+' -i $DATA_PATH/'+ \
              str(bs)+' -m $MO_MODELS_PATH/'+model[topology]+' -d CPU -api async -nireq '+str(ns)+\
              ' -niter 100 -l $SAMPLES_PATH/intel64/Release/lib/libcpu_extension.so &>'+log_file)
      print("echo 'Waiting for "+str(ns)+"-streams to finish'")
      print("sleep "+sleep_time)
      print("ps -elf | grep  samples | for i in $(awk '{print $4}');do kill -9 $i; done")

    log_file = os.path.join(LOGS_PATH,topology+"_bs"+str(bs)+"_sync.log &")
    print('$SAMPLES_PATH/intel64/Release/'+executable+' -i $DATA_PATH/'+ \
            str(bs)+' -m $MO_MODELS_PATH/'+model[topology]+' -d CPU -api sync -niter 100 -l $SAMPLES_PATH/intel64/Release/lib/libcpu_extension.so &>'+log_file)
    print("echo 'Waiting for inference to finish'")
    print("sleep 11s ")
    print("ps -elf | grep  samples | for i in $(awk '{print $4}');do kill -9 $i; done")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--cpu", default="skl6148", help="SKU name")
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
