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
MODEL_COMMON_PATH = "/home/vinod/dldt/object_detection/"
SAMPLES_COMMON_PATH = "/home/vinod/dldt/samples/intel64/Release/"

models =  {\
    "MASK_RCNN_RES50": 
    [ MODEL_COMMON_PATH + "mask_rcnn_resnet50_atrous_coco_2018_01_28/frozen_inference_graph.xml", 
      SAMPLES_COMMON_PATH + "mask_rcnn_demo"],
    "MASK_RCNN_RES101": 
    [ MODEL_COMMON_PATH + "mask_rcnn_resnet101_atrous_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "mask_rcnn_demo"],
    "FRCNN_RES50": 
    [ MODEL_COMMON_PATH + "faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "FRCNN_RES101": 
    [ MODEL_COMMON_PATH + "faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "RFCN_RES101": 
    [ MODEL_COMMON_PATH + "rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "SSD_MOB_V2": 
    [ MODEL_COMMON_PATH + "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "I8_FRCNN_RES50": 
    [ MODEL_COMMON_PATH + "faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph_i8.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "I8_FRCNN_RES101": 
    [ MODEL_COMMON_PATH + "faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph_i8.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "I8_RFCN_RES101": 
    [ MODEL_COMMON_PATH + "rfcn_resnet101_coco_2018_01_28/frozen_inference_graph_i8.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "I8_SSD_MOB_V2": 
    [ MODEL_COMMON_PATH + "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph_i8.xml",
      SAMPLES_COMMON_PATH + "object_detection_sample_ssd"],
    "I8_MASK_RCNN_RES50": 
    [ MODEL_COMMON_PATH + "imask_rcnn_resnet50_atrous_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "mask_rcnn_demo"],
    "I8_MASK_RCNN_RES101": 
    [ MODEL_COMMON_PATH + "mask_rcnn_resnet101_atrous_coco_2018_01_28/frozen_inference_graph.xml",
      SAMPLES_COMMON_PATH + "mask_rcnn_demo"],
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

      parse = file_name.split(".")[0].split('_')

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

  for top in topology:
    for bs in topology[top]:
      for stream in topology[top][bs]:
        average_latency = sum(topology[top][bs][stream])/float(len(topology[top][bs][stream]))
        num_streams = float(stream[3:])
        fps = int(bs[2:])*1000*num_streams/average_latency
        print(top,"\t\t",bs[2:],"\t",num_streams,"\t",average_latency,"\t",fps)


def create_shell_script(args):
  """
  """

  bs = args.batch_size
  if args.cpu == "8180" or args.cpu == "8280":
    NUM_CORES = 56
    SOCK_CORES = 28
    if bs == 1:
      NUM_STREAMS = [1,2,4,8,14,28,56]
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

  elif args.cpu == "6140" or args.cpu == "6240":
    NUM_CORES = 36
    SOCK_CORES = 18
    if bs == 1:
      NUM_STREAMS = [1,2,4,6]#,9,12,18,36]
    elif bs == 6:
      NUM_STREAMS = [1,2,4,6]
    elif bs == 9:
      NUM_STREAMS = [1,2,4]
    elif bs == 18:
      NUM_STREAMS = [1,2,4]
    elif bs == 36:
      NUM_STREAMS = [1,2]

  elif args.cpu == "6154" or args.cpu == "6254":
    NUM_CORES = 36
    SOCK_CORES = 18
    if bs == 1:
      NUM_STREAMS = [1,2,4,6]#,9,12,18,36]
    elif bs == 6:
      NUM_STREAMS = [1,2,4,6]
    elif bs == 9:
      NUM_STREAMS = [1,2,4]
    elif bs == 18:
      NUM_STREAMS = [1,2,4]
    elif bs == 36:
      NUM_STREAMS = [1,2]

  elif args.cpu == "VM6254" or args.cpu == "VM6154":
    NUM_CORES = 16
    SOCK_CORES = 16
    if bs == 1:
      NUM_STREAMS = [1,2,4]
    elif bs == 8:
      NUM_STREAMS = [1,2,4]
    elif bs == 16:
      NUM_STREAMS = [1,2,4]
    elif bs == 32:
      NUM_STREAMS = [1,2]

  elif args.cpu == "4S6248":
    NUM_CORES = 80
    SOCK_CORES = 20
    if bs == 1:
      NUM_STREAMS = [1,2,4,8]
    elif bs == 8:
      NUM_STREAMS = [1,2,4,5]
    elif bs == 20:
      NUM_STREAMS = [1,2,4]
    elif bs == 40:
      NUM_STREAMS = [1,2,4]

  elif args.cpu == "6148" or args.cpu == "6248":
    NUM_CORES = 40
    SOCK_CORES = 20
    if bs == 1:
      NUM_STREAMS = [1,2,4,8]
    elif bs == 8:
      NUM_STREAMS = [1,2,4,5]
    elif bs == 20:
      NUM_STREAMS = [1,2,4]
    elif bs == 40:
      NUM_STREAMS = [1,2,4]

  print("export WKDIR=$HOME/dldt/object_detection")

  if args.topology == 'all':
    top_list = [ key for key in models]
  else:
    top_list = args.topology.split(',')

  LOGS_PATH="$WKDIR/logs"
  print("export DATA_PATH=$WKDIR/coco_test_data")
  print("export SAMPLES_PATH=$WKDIR/samples")
  print("export VINO_PATH=$HOME/intel/openvino/deployment_tools")
  print("source $VINO_PATH/../bin/setupvars.sh")


  for topology in top_list:
    sleep_time = "111s"
    for ns in NUM_STREAMS:
      cores_per_stream = NUM_CORES//ns
      for i in range(ns):
        j = i*cores_per_stream
        k = (i+1)*cores_per_stream-1

        log_file = os.path.join(LOGS_PATH,topology+"_"+str(i)+"_bs"+str(bs)+"_str"+str(ns)+".log &")

        if (j < SOCK_CORES) and (k < SOCK_CORES):
          node = 0
        elif (j >= SOCK_CORES) and (k < SOCK_CORES*2):
	        node = 1
        elif (j >= SOCK_CORES*2) and (k < SOCK_CORES*3):
          node = 2
        elif (j >= SOCK_CORES*3) and (k < SOCK_CORES*4):
          node = 3
        else:
          node = -1
        if node >= 0:
          opt = "numactl  -N "+ str(node)+" -m " + str(node)+ " -C "+ str(j) + "-" + str(k) + " "
        else:
          opt = "numactl -l -C "+ str(j) + "-" + str(k) + " "

        print( opt + models[topology][1] + ' -i $DATA_PATH/'+str(bs)+' -m ' +\
                models[topology][0]+' -nthreads '+ str(cores_per_stream) +\
                ' -d CPU -ni 100  &>'+log_file)
      print("echo 'Waiting for "+str(ns)+"-streams to finish'")
      print("sleep "+sleep_time)
      print("ps -elf | grep  samples | for i in $(awk '{print $4}');do kill -9 $i; done")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--cpu", default="6254", help="SKU name")
  parser.add_argument("--topology", default="SSD_MOB_V2", help=" topology name")
  #parser.add_argument("--fw", default="TF", help="caffe/tf")
  parser.add_argument("--batch_size", type=int, default=1, help="i Batch size")
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
      create_shell_script(args)
    else:
      print_results(args)
