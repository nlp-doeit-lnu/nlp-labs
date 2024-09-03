#!/usr/bin/bash

xhost +local:root

# вкажіть власні шляхи до папок
LOCAL_LAB_HOME="/enter/path/to/NLP-labs"
LOCAL_DATA_HOME="/enter/data/home"
LOCAL_RESULT_HOME="/enter/result/home"

LOCAL_JAR_HOME="$LOCAL_HOME/programs/jars"
LOCAL_SCRIPT_HOME=$LOCAL_HOME"/scripts"
CONT_BIN_HOME="/mnt/jars"
CONT_DATA_HOME="/mnt/data"
CONT_SCRIPT_HOME="/mnt/scripts"
CONT_RESULT_HOME="/mnt/results"

LOCAL_X11_PATH="/tmp/.X11-unix"
CONT_X11_PATH="/tmp/.X11-unix"

VIDEOCARD="/dev/dri"
WEB_PORT="5800:5800"
RDP_PORT="5900:5900"

DOCKER_IMAGE="nlp-java:latest"

if [ -z "$1" ]
then
	echo "не введено назву програми"
else
	SCRIPT=$1

	docker run --device $VIDEOCARD \
		--rm \
		-p $WEB_PORT \
		-p $RDP_PORT \
		-v $LOCAL_JARS_HOME:$CONT_JARS_HOME \
		-v $LOCAL_DATA_HOME:$CONT_DATA_HOME \
		-v $LOCAL_RESULT_HOME:$CONT_RESULT_HOME \
		-v $LOCAL_SCRIPT_HOME:$CONT_SCRIPT_HOME \
		-v $LOCAL_X11_PATH:$CONT_X11_PATH \
		-e DISPLAY=$DISPLAY \
		$DOCKER_IMAGE $CONT_SCRIPT_HOME/$SCRIPT.sh
fi

xhost -local:root
