###########################################################################
# This is a sample installation file for docker on Bare metal
# Please contact vinod.devarampati@intel.com for any clarifications
# Installation instructions
# 1. Asumes that you are using intel CPUs
# 2. Latest version of CentOS/RHEL is installed (preferably minimal install)  
# 	- CentOS 7.4 (1708) is recommended
# 3. This installation requires interenet connection.
# 4. Please set the necessary proxy settings before installation
##########################################################################

# remove existing if any
sudo yum remove docker docker-common docker-selinux  docker-engine
# install pre-requisites
sudo yum install -y yum-utils   device-mapper-persistent-data   lvm2
#add repo for docker
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum-config-manager --enable docker-ce-edge
sudo yum-config-manager --enable docker-ce-test
sudo yum install docker-ce
sudo systemctl enable docker
# proxy setup
mkdir -p /etc/systemd/system/docker.service.d
# proxy settings
# add below two lines to  /etc/systemd/system/docker.service.d/http-proxy.conf  => 
# [Service]
# Environment="HTTP_PROXY=http:<site>:<port>/"
# add below two lines to  /etc/systemd/system/docker.service.d/https-proxy.conf  => 
# [Service]
# Environment="HTTPS_PROXY=https:<site>:<port>/"
#systemctl show --property=Environment docker
#add other users to docker sudo
sudo groupadd docker
sudo usermod -aG docker $USER
#run test
sudo  docker run hello-world
