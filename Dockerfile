FROM jupyter/base-notebook:b020a0cf3b96

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir notebook
RUN pip install --no-cache-dir jupyterhub
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.1-linux-x86_64.tar.gz && \
    tar -xvzf julia-1.7.1-linux-x86_64.tar.gz && \
    mv julia-1.7.1 /opt/ && \
    ln -s /opt/julia-1.7.1/bin/julia /usr/local/bin/julia && \
    rm julia-1.7.1-linux-x86_64.tar.gz

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}/MLCourse
USER root
RUN chown -R ${NB_USER} ${HOME}
USER ${NB_USER}

ENV USER_HOME_DIR /home/${NB_USER}
ENV JULIA_PROJECT ${USER_HOME_DIR}
ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/.julia
ENV JULIA_PKG_DEVDIR=${USER_HOME_DIR}
# RUN julia --project=${USER_HOME_DIR}/MLCourse -e "import Pkg; Pkg.instantiate();"
# Pkg.precompile(); using MLCourse; MLCourse.create_sysimage()"

WORKDIR ${USER_HOME_DIR}/MLCourse
RUN jupyter labextension install @jupyterlab/server-proxy && \
    jupyter lab build && \
    jupyter lab clean && \
    pip install . --no-cache-dir && \
    rm -rf ~/.cache
