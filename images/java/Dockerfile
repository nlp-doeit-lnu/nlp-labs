#FROM jlesage/baseimage-gui:ubuntu-20.04

ARG BASE=ubuntu
ARG TAG=20.04

FROM $BASE:$TAG

# installation of preliminary libraries
RUN apt update \
    && apt install -y curl \ 
     libgtk2.0-0 \
     libxtst6 \
     unzip

# JRE-18 and JavaFX installation from OpenJDK
RUN curl -O https://download.java.net/java/GA/jdk18/43f95e8614114aeaa8e8a5fcf20a682d/36/GPL/openjdk-18_linux-x64_bin.tar.gz \
    && curl -O https://download2.gluonhq.com/openjfx/17.0.12/openjfx-17.0.12_linux-x64_bin-sdk.zip \
    && tar xvf openjdk-18_linux-x64_bin.tar.gz \
    && rm openjdk-18_linux-x64_bin.tar.gz \
    && mv jdk-18 /opt/ \
    && unzip openjfx-17.0.12_linux-x64_bin-sdk.zip \
    && rm openjfx-17.0.12_linux-x64_bin-sdk.zip \
    && mv javafx-sdk-17.0.12 /opt/

ENV JAVA_HOME=/opt/jdk-18 \
    PATH=$PATH:$JAVA_HOME/bin

# COPY entrypoint.sh /usr/bin/entrypoint
# ENTRYPOINT ["/usr/bin/entrypoint"]
