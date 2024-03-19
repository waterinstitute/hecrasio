LABEL authors = "Mark Bartlett"
LABEL maintainer = "bartlett <mbartlett@thewaterinstitute.org>"

ARG conda_env=projectEnvName

USER root

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --no-install-suggests \
    gcc &&\
    apt-get autoremove && \
    apt-get clean

USER $NB_USER

COPY --chown=${NB_UID}:${NB_GID} "${conda_env}.yml" "/home/${NB_USER}/tmp/"

RUN cd "/home/${NB_USER}/tmp/" && \
     mamba env create -p "${CONDA_DIR}/envs/${conda_env}" -f "${conda_env}.yml" && \
     mamba clean --all -f -y

RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m ipykernel install --user --name="${conda_env}" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly
