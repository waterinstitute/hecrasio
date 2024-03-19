# Docker

Docker is an open platform for developing, shipping, and running applications. Docker provides the ability to package and run an application in a loosely isolated environment called a container. The Docker container allows for multiple users to consistently setup and execute script. The container isolation and security also allows for running multiple containers simultaneously on a given host (e.g., computer). Containers are lightweight and contain everything needed to run the application, so it does not rely on what is currently installed on the host. Docker containers are orchestrated localled with docker-compose.

Docker-compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application’s services. Then, with a single command, you create and start all the services from your configuration.

# Getting Started

Before getting started, for windows install Docker desktop and the windows sublinux system version 2 (for Windows 10 only), and for linux, install docker and docker-compose; see the Appendix at the end of this document.

## STEP 1 - Creating the Dockerfile

(If the Dockerfile already exists, skip this step)

Create a Dockerfile, if one doesn't exist. Typically, start the Dockerfile from a base image. A decent starting image, as well as templates for the Dockerfile in general, may be found at

<https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html>

Dockerfiles often start from images that are avaible in publicly open registries such as Docker Hub (<https://hub.docker.com/search?type=image>); however, many time an image may start with an image stored in a private registry.

### Logging into Private Registries

Note that using the develop service requires that one login into our private container registry (see Appendix at the end of this readme file).

### Installing pip and conda dependencies

For pip dependencies, the approprietely Docker files script is

```
COPY --chown=${NB_UID}:${NB_GID} docker/requirements_pip_develop.txt /tmp/

RUN pip install --requirement /tmp/requirements_pip_develop.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
```

while for conda dependencies, packages are installed via

```
COPY --chown=${NB_UID}:${NB_GID} docker/requirements_conda_develop.txt /tmp/

RUN conda install --yes --file /tmp/requirements_conda_develop.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

```

Note that pip dependencies may be installed by bypassing ssl certification with the script

`RUN pip install package_name --trusted-host pypi.org --trusted-host files.pythonhosted.org`

### Project Specifics

For this project, the container is built upon the jupyter/scipy-notebook image, which contains

* Everything in jupyter/minimal-notebook and its ancestor images
* dask, pandas, numexpr, matplotlib, scipy, seaborn, scikit-learn, scikit-image, sympy, cython, patsy, statsmodel, cloudpickle, dill, numba, bokeh, sqlalchemy, hdf5, vincent, beautifulsoup, protobuf, xlrd, bottleneck, and pytables packages
* ipywidgets and ipympl for interactive visualizations and plots in Python notebooks
* Facets for visualizing machine learning datasets.

In addition, this container is loaded with the latest version of GRASS GIS.

## STEP 2 - Building the Container with Docker Compose

(If the Dockerfile has already been tested skip this step and proceed to Step 4)

Docker-compose orchestrates the creation of the Docker container from the Docker file. It also is a tool for defining and running multi-container Docker applications (simultaneously) that share the same network.  Docker compose executes instructutions found in the (docker-compose) YAML file that configures the container application’s services. A single command (on the command line or through bash) creates and starts all the services from your configuration.

### Yaml file Configuration

Before running the container, open the docker-compose.yml file and  examine if folder volumes on the container map to the correct local folders on one's computer.  Note under the 'volumes' section of the docker-compose file, the folder location on the local computer is given before the colon while the location in the container is given after the colon. Just change the location on the computer (text before the colon), as desired. Also note the port over which the Jupyter notebook is accessed (here it set as 8080). Change as necessary.

### Running Docker Compose

#### From VS Code

(Install the required Extensions; see the Appendix in the 'src' folder README file)

From VS Code, right click on the docker-compose.yml file, and select on 'Compose Up' (if building all services), or 'Compose Up - Select Services'. Then select the services to run from the top of the VS Code screen, e.g.,

![alt text for screen readers](./README_images/compose_select_service.png "Running Docker Compose")

Once the services are running, open any Jupyter notebook within the project, and select the icon 'Jupyter Server', i.e.,

![alt text for screen readers](./README_images/jupyter_server_location.png "Jupyter Server Connection")

Upon clicking the Jupyter Server icon, a prompt will appear at the top of VS Code for the address to the Jupyter Server, i.e.,

![alt text for screen readers](./README_images/connecting_jupyter.png "Connecting to Jupyter Kernel")

To connect to the running Docker container, select 'Existing' and then insert the web address to the Jupyter Server. For a locally run Docker image, the address will appear as circled, where the port (e.g., 8080) matches the port mapping from the docker-compose yaml file.

#### From the Command Line

To use docker-compose to run the container:

* Open the command line prompt or bash. When using bash, the docker-compose command may need to be prefaced by the `sudo`.
* Change directory to the project folder with the docker-compose.yml file.
* Run the command `docker-compose up --build my-service-name` to run the docker container.

The variable `my-service-name` is found in the docker compose file as the next level directly under `services`. If a service is not listed after the command `docker-compose up --build` then the docker-compose will buid all services at once.

##### Accessing the Container

Once docker-compose has completed the command line (or bash) will display a link with an access token for accesing the Jupyter notebook. Depending on the installations, other services (such as nteract <https://nteract.io/>) may be available. Generally, using a web browser, a notebook environment is acccessed via one of the following links

* Jupyter Lab -- <http://localhost:8080/lab>
* Jupyter Notebook -- <http://localhost:8080>
* Nteract Notebook -- <http://localhost:8080/nteract>

Note that the port (here given as 8080) depends on the settings in the dockerfile. If the connection through the web browser fails, please check the port setting in the docker-comopose.yml file. For more information on Nteract, please refer to <https://nteract.io/>. Nteract was developed by Netflix <https://netflixtechblog.com/notebook-innovation-591ee3221233>

##### Rebuilding Running Container

When updating a running Docker container with installations (e.g., GRASSS GIS) or pipe and conda packages:

* Stop running the docker container with control + c
* Execute the command 'docker-compose down' from the folder where the docker-compose file resides
* Edit the requirements in the requirements.txt file or Dockerfile
* Rebuild the container with docker-compose up

#### Project Specifics

This project consists of two services:

* test - for testing the build of the image that is pushed to the Containter registry via the pipeline.
* develop - for interating the batch processing code for creating machine learning features, which pull down the Docker image from the Container Registry.

Initially use the 'test' service to check the Dockerfile guild. Thereafter, use the 'develop' service to pull the stable image (with a few packages installations that aid development). Since the 'develop' service starts from an image in the  container registry, it requires that one login into the registry following the directions in the Appendix.

## Step 3 - Store the Container in the  Registry

Once the main Dockerfile has been tested and verififed (e.g., using the 'test' service of docker-compose), the container typically should be pushed to a private registry, so the environment may be used at scale or for further development of script. The Docker image is pushed to the container registry by initiating the pipeline (called 'Docker Image for ACR') to create the Docker image for the project. The pipeline does the following steps:

* Interprets the Dockerfile to create an Image with Grass GIS, as well as with the conda and pip dependencies listed in the text files.
* Pushes the newly built image to the project Container registry with the repository name of 'grass-gis-processing' with a tag.

The best practice is to tag the image with the date (e.g., 2021-01-13 for January 13th 2021). Then test the image in the regsitry to make sure it doesn't break existing processes (typically with a unit test). Once all tests are passed, push the container to the registry with the tag 'latest', which is the tag referenced by ongoing processes.

## Step 4 - Pull the Docker Image from the  Registry

(If Dockerfile.develop already exists, just execute this file with Docker-compose for this step)

For development purposes, create a second Dockerfile (e.g., Dockerfile.develop). This Dockerfile should start from the image pushed to the  container registry (i.e., the image based on the original Dockerfile). For this new Dockerfile (for development), install pip and conda dependencies (following Step 1) that aid development, but which are not necessary for running the process at scale. Initiate this new development Dockerfile, by creating an additional service in the docker-compose file (e.g., called 'develop'). Run this new service following the docker-compose guidance given in Step 2.

# Appendix

## Required Software

For a windows environment, Docker requires the following software:

* Docker desktop - <https://www.docker.com/products/docker-desktop>
* Windows sub-Linux SYstem - <https://docs.microsoft.com/en-us/windows/wsl/install-win10>

Note that this only requires completely steps up to and including Step 5. No need to install a Linux distribution.

## Logging into an Container Registry

Unless using a publicly available image (e.g., Docker Hub), one needs to login into private container registries. To login into a container register, open the command line or bash and use the following script:

`docker login some-registry-login-server-address.io`.

When using bash, the `docker login` command may need to be `sudo docker login`. After initiating the docker login, the command line will prompt for the username and password to the container registery, i.e.,

```
$ docker login some-registry-login-server-address.io 
....
Username: my-user-name
Password:
```

For container registries, the username and password are found by browsing to the resource and clicking on access keys:

## Jupyter Lab/Notebook Internet Connectivity

If connecting to the internet with Jupyter or equivalent to download information, one first has to execute the following lines of script

* import ssl
* ssl._create_default_https_context = ssl._create_unverified_context

<https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org>

## Issues/Problems with Solutions

### WSL2 Virtual Disk (ext4.vhdx) is too Large

Docker Desktop for Windows uses WSL to manage all your images and container files and keeps them in a private virtual hard drive (VHDX) called ext4.vhdx. It's usually in C:\Users\YOUR-USER-NAME\AppData\Local\Docker\wsl\data. A significant amount of storage may be relacimed (after pruning images, etc) with Optimize-Vhd under an administrator PowerShell shell/prompt. After changing directory to where the ext4.vhdx file is located run the following command in an administrator PowerShell prompt:

```

optimize-vhd -Path .\ext4.vhdx -Mode full
```

Occassionally, with the previous command, the following error will occur: 'Hyper-V encountered an error trying to access an object on computer ‘localhost’ because the object was not found. The object might have been deleted. Verify that the Virtual Machine Management service on the computer is running. If the service is running, try to perform the task again by using Run as Administrator.' In such a case, run the following code from an administrator command prompt or PowerShell:

```

MOFCOMP %SYSTEMROOT%\System32\WindowsVirtualization.V2.mof
```

This last command recompiles the Microsoft Object Format (MOF). Subsequently, run the command to optimize the size of the ext4.vhdx file.

### Docker-Compose resources (CPUs and Memory)

The execution of script often will fail when the resources allocated in docker-compose (under the resources header) are greater than the resources available on the computer on which Docker is running.