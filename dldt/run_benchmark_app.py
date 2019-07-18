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
from operator import itemgetter
MODEL_OBJ_PATH = "/home/vinod/dldt/object_detection/"
MODEL_CLS_PATH = "/home/vinod/dldt/image_classification/frozen/"
SAMPLES_COMMON_PATH = "/home/vinod/dldt/samples/intel64/Release/"

models =  {\
    "MASK_RCNN_RES50": 
    [ MODEL_OBJ_PATH + "mask_rcnn_resnet50_atrous_coco_2018_01_28/frozen_inference_graph.xml", 
      SAMPLES_COMMON_PATH + "mask_rcnn_demo"],
    "MASK_RCNN_RES101": 
    [ MODEL_OBJ_PATH + "mask_rcnn_resnet101_atrous_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "mask_rcnn_demo"],
    "FRCNN_RES50": 
    [ MODEL_OBJ_PATH + "faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "FRCNN_RES101": 
    [ MODEL_OBJ_PATH + "faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "RFCN_RES101": 
    [ MODEL_OBJ_PATH + "rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "SSD_MOB_V2": 
    [ MODEL_OBJ_PATH + "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "I8_FRCNN_RES50": 
    [ MODEL_OBJ_PATH + "faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph_i8.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "I8_FRCNN_RES101": 
    [ MODEL_OBJ_PATH + "faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph_i8.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "I8_RFCN_RES101": 
    [ MODEL_OBJ_PATH + "rfcn_resnet101_coco_2018_01_28/frozen_inference_graph_i8.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "I8_SSD_MOB_V2": 
    [ MODEL_OBJ_PATH + "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph_i8.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "I8_MASK_RCNN_RES50": 
    [ MODEL_OBJ_PATH + "imask_rcnn_resnet50_atrous_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "mask_rcnn_demo"],
    "I8_MASK_RCNN_RES101": 
    [ MODEL_OBJ_PATH + "mask_rcnn_resnet101_atrous_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "mask_rcnn_demo"],
    "INCEPTION_V3": 
    [ MODEL_CLS_PATH + "frozen_inception_v3.xml", 
      SAMPLES_COMMON_PATH + "classification_sample"],
    "INCEPTION_V4": 
    [ MODEL_CLS_PATH + "frozen_inception_v4",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "RESNET_V1_50": 
    [ MODEL_CLS_PATH + "frozen_resnet_v1_50.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "RESNET_V1_101": 
    [ MODEL_CLS_PATH + "frozen_resnet_v1_101.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "RESNET_V1_152": 
    [ MODEL_CLS_PATH + "frozen_resnet_v1_152.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "VGG_16": 
    [ MODEL_CLS_PATH + "frozen_vgg_16.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "VGG_19": 
    [ MODEL_CLS_PATH + "frozen_vgg_19.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "I8_INCEPTION_V3": 
    [ MODEL_CLS_PATH + "frozen_inception_v3_i8.xml", 
      SAMPLES_COMMON_PATH + "classification_sample"],
    "I8_INCEPTION_V4": 
    [ MODEL_CLS_PATH + "frozen_inception_v4_i8.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "I8_RESNET_V1_50": 
    [ MODEL_CLS_PATH + "frozen_resnet_v1_50_i8.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "I8_RESNET_V1_101": 
    [ MODEL_CLS_PATH + "frozen_resnet_v1_101_i8.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "I8_RESNET_V1_152": 
    [ MODEL_CLS_PATH + "frozen_resnet_v1_152_i8.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "I8_VGG_16": 
    [ MODEL_CLS_PATH + "frozen_vgg_16_i8.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    "I8_VGG_19": 
    [ MODEL_CLS_PATH + "frozen_vgg_19_i8.xml",
      SAMPLES_COMMON_PATH + "classification_sample"],
    }

CPU_INFO = { "6240" : "Intel(R) Xeon(R) Gold 6240 CPU @ 2.60 GHz 18 Cores ",
             "6248" : "Intel(R) Xeon(R) Gold 6248 CPU @ 2.50 GHz 20 Cores ",
             "6254" : "Intel(R) Xeon(R) Gold 6254 CPU @ 3.10 GHz 18 cores ",
             "8268" : "Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90 GHz 24 cores ",
             "8280" : "Intel(R) Xeon(R) Platinum 8280 CPU @ 2.70GHz 22 cores ",
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

      bs = parse[-2]
      stream = parse[-1]
      top = parse[0]
      for name in parse[1:-3]:
        top = top+'_'+name

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

  with open("/tmp/tmp.csv","w") as fp:
    fp.write("OpenVINO R1.1 INFERENCE PERFORMANCE ON TF OBJECT DETECTION TOPOLOGIES\n")
    fp.write(CPU_INFO[args.cpu]+"\n")  
    fp.write("Dataset : IMAGENET\n\n")
    fp.write("Topology,Batch Size, Streams, Latency, Throughput, Precision\n")

    for top in topology:
      for bs in topology[top]:
        for stream in topology[top][bs]:
          average_latency = sum(topology[top][bs][stream])/float(len(topology[top][bs][stream]))
          num_streams = stream[3:]
          fps = int(bs[2:])*1000*float(num_streams)/average_latency

          if "I8" in top:
            precision = "INT8"
          else:
            precision = "FP32"

          fp.write(top + "," + bs[2:] + "," + num_streams + "," + str(average_latency) +  "," + str(fps) + "," + precision + "\n")
  with open("/tmp/tmp.csv") as fp:
    lines = fp.readlines()
    split_lines = [line.split(",") for line in lines[5:]]
    convert_lines = []
    for line in split_lines:
      if "I8" in line[0]:
        line[0] = line[0][3:]
      line[1] = int(line[1])
      line[2] = int(line[2])
      line[3] = "{0:.2f}".format(float(line[3]))
      line[4] = "{0:.2f}".format(float(line[4]))
      convert_lines.append(line)

    sorted_lines = sorted(convert_lines, key=itemgetter(0,5,1,4))
    bench_name = "bench_" + args.cpu + "_classifcation.csv"

    with open(bench_name,"w") as fp1:
      fp1.write(lines[0])
      fp1.write(lines[1])
      fp1.write(lines[2])
      fp1.write(lines[3])
      fp1.write(lines[4])
      for line in sorted_lines:
        line[1] = str(line[1]) + "-" + str(line[2])
        line[2] = str(line[3])
        line[3] = str(line[4])
        line[4] = str(line[5])
        fp1.write(','.join(line[:5]))

def create_shell_script(args):
  """
  """

  bs = args.batch_size

    # 1,2,4,8,12,16,24,32,48,64,96,128
  if args.cpu == "8268" or args.cpu == "8168":
    NUM_CORES = 48
    SOCK_CORES = 24
    if bs == 1:
      NUM_STREAMS = [1,2,4,8,12]
    elif bs == 2:
      NUM_STREAMS = [1,2,4,8,12]
    elif bs == 4:
      NUM_STREAMS = [1,2,4,8]
    elif bs == 8:
      NUM_STREAMS = [1,2,4,8]
    elif bs == 12:
      NUM_STREAMS = [1,2,4,8]
    elif bs == 16:
      NUM_STREAMS = [1,2,4]
    elif bs == 24:
      NUM_STREAMS = [1,2,4]
    elif bs == 32:
      NUM_STREAMS = [1,2,4]
    elif bs == 48:
      NUM_STREAMS = [1,2,4]
    elif bs == 64:
      NUM_STREAMS = [1,2,4]
    elif bs == 96:
      NUM_STREAMS = [1,2,4]
    elif bs == 128:
      NUM_STREAMS = [1,2,4]

  # 1,2,4,7,8,14,16,28,32,56,64,112,128
  if args.cpu == "8280" or args.cpu == "8180":
    NUM_CORES = 56
    SOCK_CORES = 28
    if bs == 1:
      NUM_STREAMS = [1,2,4,8,14]
    elif bs == 2:
      NUM_STREAMS = [1,2,4,8,14]
    elif bs == 4:
      NUM_STREAMS = [1,2,4,8]
    elif bs == 7:
      NUM_STREAMS = [1,2,4,8]
    elif bs == 8:
      NUM_STREAMS = [1,2,4,8]
    elif bs == 14:
      NUM_STREAMS = [1,2,4,8]
    elif bs == 16:
      NUM_STREAMS = [1,2,4,8]
    elif bs == 28:
      NUM_STREAMS = [1,2,4]
    elif bs == 32:
      NUM_STREAMS = [1,2,4]
    elif bs == 56:
      NUM_STREAMS = [1,2,4]
    elif bs == 64:
      NUM_STREAMS = [1,2,4]
    elif bs == 112:
      NUM_STREAMS = [1,2,4]
    elif bs == 128:
      NUM_STREAMS = [1,2,4]

  # 1,2,4,6,8,9,16,18,32,36,64,72,128
  elif args.cpu == "6140" or args.cpu == "6240" or args.cpu == "6154" or args.cpu == "6254":
    NUM_CORES = 36
    SOCK_CORES = 18
    if bs == 1:
      NUM_STREAMS = [1,2,4,6,9,12,18,36]
    elif bs == 2:
      NUM_STREAMS = [1,2,4,6,9,12,18]
    elif bs == 4:
      NUM_STREAMS = [1,2,4,6,9,12,18]
    elif bs == 6:
      NUM_STREAMS = [1,2,4,6,9,12]
    elif bs == 8:
      NUM_STREAMS = [1,2,4,6,9]
    elif bs == 9:
      NUM_STREAMS = [1,2,4,6,12]
    elif bs == 16:
      NUM_STREAMS = [1,2,4,6,9]
    elif bs == 18:
      NUM_STREAMS = [1,2,4,6,9]
    elif bs == 32:
      NUM_STREAMS = [1,2,4,6]
    elif bs == 36:
      NUM_STREAMS = [1,2,4,6]
    elif bs == 64:
      NUM_STREAMS = [1,2,4,6]
    elif bs == 72:
      NUM_STREAMS = [1,2,4]
    elif bs == 128:
      NUM_STREAMS = [1,2,4]

  # 1,2,4,8,20,40,80
  elif args.cpu == "6148" or args.cpu == "6248":
    NUM_CORES = 40
    SOCK_CORES = 20
    if bs == 1:
      NUM_STREAMS = [1,2,4,8,10]
    elif bs == 2:
      NUM_STREAMS = [1,2,4,10]
    elif bs == 4:
      NUM_STREAMS = [1,2,4,5,10]
    elif bs == 8:
      NUM_STREAMS = [1,2,4,5]
    elif bs == 16:
      NUM_STREAMS = [1,2,4,5]
    elif bs == 20:
      NUM_STREAMS = [1,2,4,5]
    elif bs == 32:
      NUM_STREAMS = [1,2,4,5]
    elif bs == 40:
      NUM_STREAMS = [1,2,4]
    elif bs == 64:
      NUM_STREAMS = [1,2,4]
    elif bs == 80:
      NUM_STREAMS = [1,2,4]
    elif bs == 128:
      NUM_STREAMS = [1,2,4]


  if args.topology == 'all':
    top_list = [ key for key in models]
  else:
    top_list = args.topology.split(',')

  LOGS_PATH="$WKDIR/logs"


  for topology in top_list:
    sleep_time = str(100)+str(bs*10)+'s'
    for ns in NUM_STREAMS:
      cores_per_stream = NUM_CORES//ns
      log_file = os.path.join(LOGS_PATH,topology+"_bs"+str(bs)+"_async_req"+str(ns)+".log ")
      print("echo 'Waiting for async "+str(ns)+"-streams to finish'")
      print('numactl -l $SAMPLES_PATH/intel64/Release/benchmark_app -i $DATA_PATH/'+ \
              ' -m '+models[topology][0]+' -d CPU -api async -nireq '+ str(ns) + ' -b ' + str(bs) + \
              ' -niter 100 -l $SAMPLES_PATH/intel64/Release/lib/libcpu_extension.so &>' + log_file)
      #print("sleep "+sleep_time)

    log_file = os.path.join(LOGS_PATH,topology+"_bs"+str(bs)+"_sync.log ")
    print("echo 'Waiting for sync inference to finish'")
    print('numactl -l $SAMPLES_PATH/intel64/Release/benchmark_app -i $DATA_PATH/'+ ' -b' + \
            str(bs)+' -m '+ models[topology][0] +' -d CPU -api sync -niter 100 -l $SAMPLES_PATH/intel64/Release/lib/libcpu_extension.so &>'+log_file)
    #print("sleep 11s ")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--cpu", type=str, default="6254", help="SKU name")
  parser.add_argument("--topology", default="SSD_MOB_V2", help="topology name")
  parser.add_argument("--batch_size", type=str, default='1', help="Batch size")
  parser.add_argument("--mode", type=str, default="exe", help="exe/log")
  parser.add_argument("--data_type", type=str, default="f32", help="f32/f16/i8")
  parser.add_argument("--log_dir", type=str, default="./", help="logs directory")
  parser.add_argument("--display", type=bool, default=False, help="Display supported topologies")
  args = parser.parse_args()
  if args.display:
    for key in models:
      print(key,end = ",")
  else:
    if args.mode == "exe":
      batch_sizes = args.batch_size.split(",")
      print("export DATA_PATH=$WKDIR/coco_test_data")
      print("export WKDIR=$HOME/dldt/object_detection")
      print("export SAMPLES_PATH=$WKDIR/../samples")
      print("export VINO_PATH=$HOME/intel/openvino/deployment_tools")
      print("source $VINO_PATH/../bin/setupvars.sh")
      print("ps -elf | grep  samples | for i in $(awk '{print $4}');do kill -9 $i; done")
      for size in batch_sizes:
        args.batch_size = int(size)
        create_shell_script(args)
    else:
      print_results(args)
