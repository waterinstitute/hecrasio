# HECRASIO (HEC-RAS Input Output)


## Description
The respository hecrasio is a collection of tools to read results from HEC-RAS providing quality assurance and control (QA/QC) of one or more notebooks. HEC-RAS input and output is processed and displayed by running the [QA/QC - LWI](https://github.com/user/repo/blob/branch/other_file.md) Jupyter notebook. Running the notebook should be done within a Docker container created with the [Dockerfile-hecrasio](https://github.com/user/repo/blob/branch/other_file.md). The Docker container is created by running the [docker-compose](https://github.com/user/repo/blob/branch/other_file.md) file. Once the docker container is runnning, the Jupyter notebook [QA/QC - LWI](https://github.com/user/repo/blob/branch/other_file.md) is accessed by going to [localhost:8889](ocalhost:8889/lab/tree/jovyan/app/notebooks) and entering the password 'LWI'.


## Data
For using the QA/QC notebook in examining HEC-RAS data, the HEC-RAS HDF should be uploaded to the data/external folder.

## Project organization

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── config             <- Configuration files for the project.
    ├── build              <- Files for building environments
    │   ├── docker         <- Docker-compose, Dockerfile, requirements, etc. for the project.
    │   ├── k8s            <- Kubernetes files for the project
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │  
    │   │── deployment     <- Scripts to deploy the model as a service
    │   │   └── deploy_local.py
    │   │   └── deploy.py    
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── LICENSE
