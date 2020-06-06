distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
# install nvidia-container-toolkit
sudo yum install -y nvidia-container-toolkit
sudo systemctl restart docker
# run docker with gpu options
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
