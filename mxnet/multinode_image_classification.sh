###################################################################################
# Benchmarks for multinode image classification topologies with ImageNet Data
# Owner : Vinod.devarampati@intel.com
# Description : 
#   *hostfile* should contain the host name on which multinode training should run.
#   Kill the process in all nodes after completion of a topology
#   Read and modify or uncomment some of the lines depending on the need
#   All the nodes should have the same file structure. More info at below link 
#     -https://mxnet.incubator.apache.org/faq/multi_devices.html
###################################################################################
#!/bin/bash
# update the MX_PATH
export MX_PATH=~/mxnet_mkldnn
# update .rec files path
export IMG_DATA_PATH=~/mxnet_data
export MKLDNN_ROOT=$MX_PATH/external/mkldnn
export LD_LIBRARY_PATH=$MKLDNN_ROOT/lib:$LD_LIBRARY_PATH
## Following values are for SKL6148 and should be changed for other SKUs ##
# make DUMMY=1 for DUMMY Data tests and change the OMP_NUM_THREADS
export DUMMY=0 # ;export OMP_NUM_THREADS=40;
export OMP_NUM_THREADS=79;
export ENDPROC=79 #max threads-1
export KMP_AFFINITY=granularity=thread,proclist=[0-$ENDPROC],explicit,verbose;
export PS_VERBOSE=1;
# uncomment and change to required IF
#export DMLC_INTERFACE=eno1;
echo "Threads=$OMP_NUM_THREADS. Affinity=$KMP_AFFINITY. Make sure that hyper-threading is ON"
echo "Executing Mxnet benchmarks"


python $MX_PATH/tools/launch.py -n 4 -s 1 -H ~/hostfile --launcher ssh  numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network googlenet --num-layers 0 --batch-size 64  --num-epochs 1 --kv-store dist_device_sync  --image-shape 3,256,256 --data-nthreads 2  --data-train $IMG_DATA_PATH/256_data/imagenet10_train.rec  --benchmark $DUMMY --disp-batches 10 &>googlenet.log &

python $MX_PATH/tools/launch.py -n 4 -s 1 -H ~/hostfile --launcher ssh  numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network vgg --num-layers 16 --batch-size 64  --num-epochs 1 --kv-store dist_device_sync  --image-shape 3,224,224 --data-nthreads 2  --data-train $IMG_DATA_PATH/256_data/imagenet10_train.rec  --benchmark $DUMMY --disp-batches 10 &>vgg16.log &

python $MX_PATH/tools/launch.py -n 4 -s 1 -H ~/hostfile --launcher ssh  numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network vgg --num-layers 19 --batch-size 64  --num-epochs 1 --kv-store dist_device_sync  --image-shape 3,224,224 --data-nthreads 2  --data-train $IMG_DATA_PATH/256_data/imagenet10_train.rec  --benchmark $DUMMY --disp-batches 10 &>vgg19.log &

python $MX_PATH/tools/launch.py -n 4 -s 1 -H ~/hostfile --launcher ssh  numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network inception-v3 --num-layers 0 --batch-size 64  --num-epochs 1 --kv-store dist_device_sync  --image-shape 3,299,299 --data-nthreads 2  --data-train $IMG_DATA_PATH/299_data/imagenet_train.rec  --benchmark $DUMMY --disp-batches 10 &>inception3.log &

python $MX_PATH/tools/launch.py -n 4 -s 1 -H ~/hostfile --launcher ssh  numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network inception-v4 --num-layers 0 --batch-size 64  --num-epochs 1 --kv-store dist_device_sync  --image-shape 3,299,299 --data-nthreads 2  --data-train $IMG_DATA_PATH/299_data/imagenet_train.rec  --benchmark $DUMMY --disp-batches 10 &>inception4.log &

python $MX_PATH//tools/launch.py -n 4 -s 1 -H ~/hostfile --launcher ssh  numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network resnet --num-layers 50 --batch-size 64  --num-epochs 1 --kv-store dist_device_sync  --image-shape 3,224,224 --data-nthreads 2  --data-train $IMG_DATA_PATH/256_data/imagenet10_train.rec  --benchmark $DUMMY --disp-batches 10 &>renet50.log &

python $MX_PATH/tools/launch.py -n 4 -s 1 -H ~/hostfile --launcher ssh  numactl -l python -u $MX_PATH/example/image-classification/train_imagenet.py   --network resnet --num-layers 101 --batch-size 64  --num-epochs 1 --kv-store dist_device_sync  --image-shape 3,224,224 --data-nthreads 2  --data-train $IMG_DATA_PATH/256_data/imagenet10_train.rec  --benchmark $DUMMY --disp-batches 10 &>resnet101.log &
