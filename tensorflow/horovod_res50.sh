# single node without MPI settings - 2S Xeon 6148

numactl -l  python -u $TF_BENCH/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --num_batches 40 --distortions=False --local_parameter_device cpu  --num_intra_threads 40 --num_inter_threads 2 --data_dir $TF_RECORDS --data_name imagenet

# Single node with MPI settings - 2S Xeon 6148 - 40 cores

export OMP_NUM_THREADS=40; export KMP_AFFINITY=granularity=fine,verbose,compact,1,0; export KMP_BLOCKTIME=1; numactl -l  python -u $TF_BENCH/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --num_batches 40 --distortions=False --local_parameter_device cpu  --num_intra_threads 40 --num_inter_threads 2 --data_dir $TF_RECORDS --data_name imagenet

# Single node with MPI settings - 2S Xeon 6148 - 20 cores

export OMP_NUM_THREADS=20; export KMP_AFFINITY=granularity=fine,verbose,compact,1,0; export KMP_BLOCKTIME=1; numactl -m 0 -N 0  python -u $TF_BENCH/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --num_batches 40 --distortions=False --local_parameter_device cpu  --num_intra_threads 20 --num_inter_threads 2 --data_dir $TF_RECORDS --data_name imagenet

# Single node with MPI settings - 2S Xeon 6148 - 10 cores

export OMP_NUM_THREADS=10; export KMP_AFFINITY="granularity=thread,proclist=[0-9,40-49],explicit,verbose"; export KMP_BLOCKTIME=1; numactl -m 0 -N 0  python -u $TF_BENCH/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --num_batches 40 --distortions=False --local_parameter_device cpu  --num_intra_threads 10 --num_inter_threads 2 --data_dir $TF_RECORDS --data_name imagenet

# Single node with MPI settings and horovod  - 2S Xeon 6148 - 20 cores

HOROVOD_FUSION_THRESHOLD=134217728 mpirun -np 2 --map-by ppr:2:socket:pe=10 -H $HOSTNAME --oversubscribe --report-bindings -x LD_LIBRARY_PATH -x HOROVOD_FUSION_THRESHOLD -x OMP_NUM_THREADS=10 -x KMP_BLOCKTIME=1  -x KMP_AFFINITY=granularity=fine,verbose,compact,1,0 python -u $TF_BENCH/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --num_batches 40 --distortions=False --local_parameter_device cpu  --num_intra_threads 10 --num_inter_threads 2 --data_dir $TF_RECORDS --data_name imagenet --variable_update horovod --horovod_device cpu

# Single node with MPI settings and horovod  - 2S Xeon 6148 - 40 cores 

HOROVOD_FUSION_THRESHOLD=134217728 mpirun -np 4 --map-by ppr:2:socket:pe=10 -H $HOSTNAME --oversubscribe --report-bindings -x LD_LIBRARY_PATH -x HOROVOD_FUSION_THRESHOLD -x OMP_NUM_THREADS=10 -x KMP_BLOCKTIME=1  -x KMP_AFFINITY=granularity=fine,verbose,compact,1,0 python -u $TF_BENCH/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --num_batches 40 --distortions=False --local_parameter_device cpu  --num_intra_threads 10 --num_inter_threads 2 --data_dir $TF_RECORDS --data_name imagenet --variable_update horovod --horovod_device cpu

# Dual node with MPI settings and horovod    - 2S Xeon 6148 - 80 cores

HOROVOD_FUSION_THRESHOLD=134217728 mpiexec -np 8 --map-by ppr:2:socket:pe=10 -hostfile ~/tf/hostfile -mca btl_tcp_if_exclude lo,enp92s0f1,eno1,eno2 -mca pml ob1 -mca btl ^openib --oversubscribe --report-bindings -x LD_LIBRARY_PATH -x HOROVOD_FUSION_THRESHOLD -x OMP_NUM_THREADS=10 -x KMP_BLOCKTIME=1  -x KMP_AFFINITY=granularity=fine,verbose,compact,1,0 python -u $TF_BENCH/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --num_batches 40 --distortions=False --local_parameter_device cpu  --num_intra_threads 10 --num_inter_threads 2 --data_dir $TF_RECORDS --data_name imagenet --variable_update horovod --horovod_device cpu
