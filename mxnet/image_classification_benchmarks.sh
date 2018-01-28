####################################################################
# Benchmarks for image classification topologies with ImageNet Data
# Owner : Vinod.devarampati@intel.com
####################################################################
#!/bin/bash
# update the MX_PATH
export MX_PATH=~/mkldnn_mxnet
# update .rec files path
export IMG_DATA_PATH=~/mxnet_data
export MKLDNN_ROOT=$MX_PATH/external/mkldnn
export LD_LIBRARY_PATH=$MKLDNN_ROOT/lib:$LD_LIBRARY_PATH
## Following values are for SKL6148 and should be changed for other SKUs ##
# make DUMMY=1 for DUMMY Data tests and change the OMP_NUM_THREADS
export DUMMY=0 # ;export OMP_NUM_THREADS=40; 
export OMP_NUM_THREADS=76;
export ENDPROC=79 #max threads-1 
export KMP_AFFINITY=granularity=thread,proclist=[0-$ENDPROC],explicit,verbose;

echo "Threads=$OMP_NUM_THREADS. Affinity=$KMP_AFFINITY. Make sure that hyper-threading is ON"
echo "Executing Mxnet benchmarks"

source ~/mx.sh; numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network googlenet --num-layers 0 --batch-size 128  --num-epochs 1 --kv-store device  --image-shape 3,256,256 --data-nthreads 2  --data-train $IMG_DATA_PATH/256_data/imagenet10_train.rec  --benchmark $DUMMY --disp-batches 10 &>googlenet.log

source ~/mx.sh; numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network vgg --num-layers 16 --batch-size 128  --num-epochs 1 --kv-store device  --image-shape 3,224,224 --data-nthreads 2  --data-train $IMG_DATA_PATH/256_data/imagenet10_train.rec  --benchmark $DUMMY --disp-batches 10 &>vgg16.log

source ~/mx.sh; numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network vgg --num-layers 19 --batch-size 128  --num-epochs 1 --kv-store device  --image-shape 3,224,224 --data-nthreads 2  --data-train $IMG_DATA_PATH/256_data/imagenet10_train.rec  --benchmark $DUMMY --disp-batches 10 &>vgg19.log

source ~/mx.sh; numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network inception-v3 --num-layers 0 --batch-size 128  --num-epochs 1 --kv-store device  --image-shape 3,299,299 --data-nthreads 2  --data-train $IMG_DATA_PATH/299_data/imagenet_train.rec  --benchmark $DUMMY --disp-batches 10 &>inception3.log

source ~/mx.sh; numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network inception-v4 --num-layers 0 --batch-size 128  --num-epochs 1 --kv-store device  --image-shape 3,299,299 --data-nthreads 2  --data-train $IMG_DATA_PATH/299_data/imagenet_train.rec  --benchmark $DUMMY --disp-batches 10 &>inception4.log

source ~/mx.sh; numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network resnet --num-layers 50 --batch-size 128  --num-epochs 1 --kv-store device  --image-shape 3,224,224 --data-nthreads 2  --data-train $IMG_DATA_PATH/256_data/imagenet10_train.rec  --benchmark $DUMMY --disp-batches 10 &>renet50.log

source ~/mx.sh; numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network resnet --num-layers 101 --batch-size 128  --num-epochs 1 --kv-store device  --image-shape 3,224,224 --data-nthreads 2  --data-train $IMG_DATA_PATH/256_data/imagenet10_train.rec  --benchmark $DUMMY --disp-batches 10 &>resnet101.log

