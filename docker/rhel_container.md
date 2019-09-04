# This section shows how to install and run RHEL UBI container
## RHEL initialization
```
subscription-manager register
subscription-manager attach
subscription-manager repos --list
subscription-manager repos --enable rhel-7-server-extras-rpms rhel-7-server-optional-rpms
```
## Buildah installaton and setup
```
sudo yum -y install buildah
sudo buildah login -u <username> -p <password>
```
## building and running docker images
```
### keep Dockerfile and dependency files with all commands in a directory. Move to the directory and give below commands
sudo buildah bud -t <image_name> .
sudo podman run -i <img_name> 
```
