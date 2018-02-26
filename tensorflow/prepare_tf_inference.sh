###########################################################################
# This is a sample script file to run TF nference benchmarks with 
# mkldnn library on Intel CPUs
# Please contact vinod.devarampati@intel.com for any clarifications
# Instructions to fill the variable are given in comments
# make sure you are using TF1.5 or later to use these scripts
############################################################################


#Point to the path where you want to load all files
export WKDIR=~/tf_inference_demo
#Point to the path where you want to load all images 
export IMAGE_DIR=~/jpeg

mkdir -p $WKDIR
cd $WKDIR
# download all trained .pb files
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
tar -xvf vgg_19_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvf inception_v3_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
tar -xvf inception_v4_2016_09_09.tar.gz
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xvf resnet_v1_50_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xvf resnet_v1_101_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
tar -xvf resnet_v1_152_2016_08_28.tar.gz
mkdir -p $WKDIR/ckpt
mv *.ckpt ckpt
rm -rf *.gz
#download the code
git clone https://github.com/tensorflow/models.git
git clone https://github.com/tensorflow/tensorflow.git
mkdir -p $WKDIR/pb
mkdir -p $WKDIR/frozen
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=vgg_16 --output_file=$WKDIR/pb/vgg_16.pb  --labels_offset=1
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=vgg_19 --output_file=$WKDIR/pb/vgg_19.pb  --labels_offset=1
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=resnet_v1_50 --output_file=$WKDIR/pb/resnet_v1_50.pb  --labels_offset=1
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=resnet_v1_101 --output_file=$WKDIR/pb/resnet_v1_101.pb  --labels_offset=1
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=resnet_v1_152 --output_file=$WKDIR/pb/resnet_v1_152.pb  --labels_offset=1
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=inception_v3 --output_file=$WKDIR/pb/inception_v3.pb
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=inception_v4 --output_file=$WKDIR/pb/inception_v4.pb
cd $WKDIR/tensorflow
bazel build tensorflow/python/tools:freeze_graph
# summarize graph helps in giving details of node names for freeze.
#bazel build tensorflow/tools/graph_transforms:summarize_graph
#bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=$WKDIR/pb/resnet_v1_101.pb
# freeze graph to load into the inference program
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/vgg_16.pb   --input_checkpoint=$WKDIR/ckpt/vgg_16.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_vgg_16.pb   --output_node_names=vgg_16/fc8/squeezed
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/vgg_19.pb   --input_checkpoint=$WKDIR/ckpt/vgg_19.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_vgg_19.pb   --output_node_names=vgg_19/fc8/squeezed
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/resnet_v1_50.pb   --input_checkpoint=$WKDIR/ckpt/resnet_v1_50.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_resenet_v1_50.pb   --output_node_names=resnet_v1_50/predictions/Reshape_1
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/resnet_v1_101.pb   --input_checkpoint=$WKDIR/ckpt/resnet_v1_101.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_resenet_v1_101.pb   --output_node_names=resnet_v1_101/predictions/Reshape_1
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/resnet_v1_152.pb   --input_checkpoint=$WKDIR/ckpt/resnet_v1_152.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_resenet_v1_152.pb   --output_node_names=resnet_v1_152/predictions/Reshape_1
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/inception_v3.pb   --input_checkpoint=$WKDIR/ckpt/inception_v3.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_inception_v3.pb   --output_node_names=InceptionV3/Predictions/Reshape_1
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/inception_v4.pb   --input_checkpoint=$WKDIR/ckpt/inception_v4.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_inception_v4.pb   --output_node_names=InceptionV4/Logits/Predictions
cd $WKDIR
#run the inference benchmarking with desired params
#export OMP_NUM_THREADS=4
#export KMP_AFFINITY="granularity=thread,proclist=[0-3,56-59],explicit,verbose"
echo "Synthetic data Benchmarking"
numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_vgg_16.pb --labels imagenet_slim_labels.txt --input_height 224 --input_width 224 --input_mean 0 --input_std 255 --input_layer input --output_layer vgg_16/fc8/squeezed --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg --data_type synthetic

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_vgg_19.pb --labels imagenet_slim_labels.txt --input_height 224 --input_width 224 --input_mean 0 --input_std 255 --input_layer input --output_layer vgg_19/fc8/squeezed --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg --data_type synthetic

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_resenet_v1_50.pb --labels imagenet_slim_labels.txt --input_height 224 --input_width 224 --input_mean 0 --input_std 255 --input_layer input --output_layer resnet_v1_50/predictions/Reshape_1 --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg --data_type synthetic

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_resenet_v1_101.pb --labels imagenet_slim_labels.txt --input_height 224 --input_width 224 --input_mean 0 --input_std 255 --input_layer input --output_layer resnet_v1_101/predictions/Reshape_1 --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg --data_type synthetic

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_resenet_v1_152.pb --labels imagenet_slim_labels.txt --input_height 224 --input_width 224 --input_mean 0 --input_std 255 --input_layer input --output_layer resnet_v1_152/predictions/Reshape_1 --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg --data_type synthetic

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_inception_v3.pb --labels imagenet_slim_labels.txt --input_height 299 --input_width 299 --input_mean 0 --input_std 255 --input_layer input --output_layer InceptionV3/Predictions/Reshape_1 --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg --data_type synthetic

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_inception_v4.pb --labels imagenet_slim_labels.txt --input_height 299 --input_width 299 --input_mean 0 --input_std 255 --input_layer input --output_layer InceptionV4/Logits/Predictions --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg --data_type synthetic


echo "ImageNet data Benchmarking"

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_vgg_16.pb --labels imagenet_slim_labels.txt --input_height 224 --input_width 224 --input_mean 0 --input_std 255 --input_layer input --output_layer vgg_16/fc8/squeezed --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg 

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_vgg_19.pb --labels imagenet_slim_labels.txt --input_height 224 --input_width 224 --input_mean 0 --input_std 255 --input_layer input --output_layer vgg_19/fc8/squeezed --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg 

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_resenet_v1_50.pb --labels imagenet_slim_labels.txt --input_height 224 --input_width 224 --input_mean 0 --input_std 255 --input_layer input --output_layer resnet_v1_50/predictions/Reshape_1 --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg 

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_resenet_v1_101.pb --labels imagenet_slim_labels.txt --input_height 224 --input_width 224 --input_mean 0 --input_std 255 --input_layer input --output_layer resnet_v1_101/predictions/Reshape_1 --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg 

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_resenet_v1_152.pb --labels imagenet_slim_labels.txt --input_height 224 --input_width 224 --input_mean 0 --input_std 255 --input_layer input --output_layer resnet_v1_152/predictions/Reshape_1 --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg 

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_inception_v3.pb --labels imagenet_slim_labels.txt --input_height 299 --input_width 299 --input_mean 0 --input_std 255 --input_layer input --output_layer InceptionV3/Predictions/Reshape_1 --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg 

numactl -l python tf_inference.py --image_dir $IMAGE_DIR --graph $WKDIR/frozen/frozen_inception_v4.pb --labels imagenet_slim_labels.txt --input_height 299 --input_width 299 --input_mean 0 --input_std 255 --input_layer input --output_layer InceptionV4/Logits/Predictions --batch_size 16 --top_accuracy 1  --benchmark 1 --image_type jpeg 
