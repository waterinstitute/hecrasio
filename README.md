# HECRASIO (HEC-RAS Input Output)


## Description
This respository (hecrasio) demonstrates how to read information from the HEC-RAS .hdf file. HEC-RAS input and output is processed and displayed by running the [QA/QC - LWI](https://github.com/waterinstitute/hecrasio/blob/main/notebooks/QAQC-LWI.ipynb) Jupyter notebook. The notebook should be run within a Docker container created with the file [Dockerfile.hecrasio](https://github.com/waterinstitute/hecrasio/blob/main/build/docker/Dockerfile.hecrasio). The Docker container is created by running the [docker-compose](https://github.com/waterinstitute/hecrasio/blob/main/build/docker/docker-compose.yml) file (with VS code or through the command line). Once the docker container is runnning, the Jupyter notebook [QA/QC - LWI](https://github.com/waterinstitute/hecrasio/blob/main/notebooks/QAQC-LWI.ipynb) is accessed by opening a web browser (e.g., Chrome), going to the web address of [localhost:8889](http://localhost:8889/lab/tree/jovyan/app/notebooks), and entering the password 'LWI'.


## Data
For using the QA/QC notebook in examining HEC-RAS data, the HEC-RAS .hdf file should be uploaded to the folder 'data/processed'.

## Project organization

    ├── build              <- Files for building environments
    │   └── docker         <- Docker-compose, Dockerfile, requirements, etc. for the project.
    │    
    ├── data
    │   └── processed      <- The folder to store HEC-RAS .hdf files for analysis with the notebook
    │
    ├── notebooks          <- LWI QA/QC notebook for HEC-RAS input/output analysis
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   └── data           <- Scripts to download or generate data
    │       └── hecrasio   <- Scripts to analyze the HEC-RAS input/output
    │
    └── LICENSE
