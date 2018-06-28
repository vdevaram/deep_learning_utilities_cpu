from __future__ import print_function
import os.path
from subprocess import call
models_tf =  {
   "vgg_16":["input", "vgg_16/fc8/squeezed", "frozen_vgg_16.pb"],
   "vgg_19":["input", "vgg_19/fc8/squeezed", "frozen_vgg_19.pb"],
   "inception_v3":["input", "InceptionV3/Predictions/Reshape_1", "frozen_inception_v3.pb"],
   "inception_v4":["input", "InceptionV4/Logits/Predictions", "frozen_inception_v4.pb"],
   "resnet_v1_50":["input", "resnet_v1_50/predictions/Reshape_1", "frozen_resnet_v1_50.pb"],
   "resnet_v1_101":["input", "resnet_v1_101/predictions/Reshape_1", "frozen_resnet_v1_101.pb"],
   "resnet_v1_152":["input", "resnet_v1_152/predictions/Reshape_1", "frozen_resnet_v1_152.pb"]
    }
GTT = "/home/vinod/gtt"
PATH_F = "/home/vinod/gtt/frozen"
PATH_O = "/home/vinod/gtt/optimized"
TF_EXE = "/home/vinod/work/tf/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph"
TF_PATH= "/home/vinod/work/tf/tensorflow/"
fp = open("transform.sh","w")
for model in models_tf:
  in_graph = models_tf[model][-1]
  in_graph = os.path.join(PATH_F, in_graph)
  out_graph = "opt_"+models_tf[model][-1] 
  out_graph = os.path.join(PATH_O,out_graph)
  in_node = models_tf[model][0]
  out_node = models_tf[model][1]
  fp.write('echo Procesing -inputGraph: '+in_graph+"outputGraph: "+out_graph+"InputNode: "+in_node+"OutputNode:"+out_node+"\n")
  fp.write((TF_EXE+" --in_graph="+in_graph+" --out_graph="+out_graph+" --inputs="+in_node+"  --outputs="+out_node+" --transforms='strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms  '"))
  #fp.write((TF_EXE+" --in_graph="+in_graph+" --out_graph="+out_graph+" --inputs="+in_node+"  --outputs="+out_node+" --transforms='strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms merge_duplicate_nodes obfuscate_names remove_device remove_control_dependencies sort_by_execution_order '"))
  fp.write("\n")
fp.close()
file_sh = os.path.join(GTT,"transform.sh")
print(file_sh)
call(["chmod","777",file_sh])
