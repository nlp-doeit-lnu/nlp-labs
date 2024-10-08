#FROM jlesage/baseimage-gui:ubuntu-20.04

ARG BASE=ubuntu
ARG TAG=latest

FROM $BASE:$TAG

# installation of preliminary libraries
RUN apt update \
    && apt install -y curl \
    pip \
    python3 \
    python3-tk \
    python3-numpy \
    python3-matplotlib \
    python3-scipy \
    python3-pandas \
    python3-openpyxl \
    python3-networkx \
    python3-regex \
    python3-nltk \
    python3-plotly

RUN pip install dash dash_bootstrap_components dash_html_components dash_core_components scikit-learn numba --break-system-packages

EXPOSE 8050
