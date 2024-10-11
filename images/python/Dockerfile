#FROM jlesage/baseimage-gui:ubuntu-20.04

ARG BASE=ubuntu
ARG TAG=20.04

FROM $BASE:$TAG

# ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# installation of preliminary libraries
RUN apt update\
    && apt install -y --no-install-recommends curl \
    pip \
    python3 \
    python3-tk

RUN pip install --no-compile --no-cache-dir dash \
    dash_bootstrap_components \
    dash_html_components \ 
    dash_core_components \
    matplotlib \
    networkx \
    nltk \
    numba \
    numpy \
    openpyxl \
    pandas \
    plotly \
    regex \
    scikit-learn \
    scipy 

EXPOSE 8050
