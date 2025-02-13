#!/bin/bash

NLP_HOME="/home/bohdan/git_projects/NLP-labs"

NLP_DATA="/home/bohdan/git_projects/NLP-dcr"
NLP_RESULTS="/home/bohdan/git_projects/NLP-dcr"

LOCAL_JARS="$NLP_HOME/programs/jars"
LOCAL_SCRIPTS="$NLP_HOME/scripts"

MOUNT_POINT="/mnt"
CONT_JARS="$MOUNT_POINT/jars"
CONT_DATA="$MOUNT_POINT/data"
CONT_SCRIPTS="$MOUNT_POINT/scripts"
CONT_RESULTS="$MOUNT_POINT/results"

LOCAL_X11_PATH="/tmp/.X11-unix"
CONT_X11_PATH="/tmp/.X11-unix"

XKEY_BIND="/home/bohdan/.nlp-docker.Xkey:/root/.Xkey:ro"

VIDEOCARD="/dev/dri"
WEB_PORT="5800:5800"
RDP_PORT="5900:5900"

DOCKER_IMAGE="nlp-java"
TAG="latest"

if [[ "$OSTYPE" == "Darwin" ]]
then
	echo "starting XQuartz"
	open -a XQuartz
fi

# xhost +local:root

docker run --rm -it -v $LOCAL_JARS:$CONT_JARS \
	-v $LOCAL_SCRIPTS:$CONT_SCRIPTS \
	-v $NLP_DATA:$CONT_DATA:ro \
	-v $NLP_RESULTS:$CONT_RESULTS \
	-v $LOCAL_X11_PATH:$CONT_X11_PATH \
	-v $XKEY_BIND \
	-e DISPLAY=$DISPLAY \
	$DOCKER_IMAGE:$TAG $CONT_SCRIPTS/entry.sh

	# xhost -local:root
