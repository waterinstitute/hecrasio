# HECRASIO (HEC-RAS Input Output)


## Description
This respository (hecrasio) demonstrates how to read information from the HEC-RAS .hdf file. HEC-RAS input and output is processed and displayed by running the [QA/QC - LWI](https://github.com/waterinstitute/hecrasio/blob/main/notebooks/QAQC-LWI.ipynb) Jupyter notebook. The notebook should be run within a Docker container created with the file [Dockerfile.hecrasio](https://github.com/waterinstitute/hecrasio/blob/main/build/docker/Dockerfile.hecrasio). The Docker container is created by running the [docker-compose](https://github.com/waterinstitute/hecrasio/blob/main/build/docker/docker-compose.yml) file (with VS code or through the command line). Once the docker container is runnning, the Jupyter notebook [QA/QC - LWI](https://github.com/waterinstitute/hecrasio/blob/main/notebooks/QAQC-LWI.ipynb) is accessed by opening a web browser (e.g., Chrome), going to the web address of [localhost:8889](http://localhost:8889/lab/tree/jovyan/app/notebooks), and entering the password 'LWI'.


## Data
For using the QA/QC notebook in examining HEC-RAS data, the HEC-RAS HDF should be uploaded to the folder 'data/processed'.

## Project organization

    ├── config             <- Configuration files for the project.
    ├── build              <- Files for building environments
    │   ├── docker         <- Docker-compose, Dockerfile, requirements, etc. for the project.
    │    
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │       └── hecrasio
    │
    └── LICENSE
