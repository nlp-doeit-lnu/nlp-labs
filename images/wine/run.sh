#!/usr/bin/bash

LOCAL_LAB_HOME="/home/bohdan/git_projects/NLP-labs"
LOCAL_DATA_HOME="/enter/data/home"
LOCAL_RESULT_HOME="/enter/result/home"

LOCAL_BIN_HOME="$LOCAL_HOME/programs/binaries"
LOCAL_SCRIPT_HOME=$LOCAL_HOME"/scripts"
CONT_BIN_HOME="/mnt/binaries"
CONT_DATA_HOME="/mnt/data"
CONT_SCRIPT_HOME="/mnt/scripts"
CONT_RESULT_HOME="/mnt/results"

if [ -z "$1" ]
then
	echo "не введено назву програми"
else

	SCRIPT=$1

	./docker-wine --cache \
		--local \
		--as-root \
		--volume="$LOCAL_BIN_HOME:$CONT_BIN_HOME" \
		--volume="$LOCAL_DATA_HOME:$CONT_DATA_HOME" \
		--volume="$LOCAL_SCRIPT_HOME:$CONT_SCRIPT_HOME" \
		--volume="$LOCAL_RESULT_HOME:$CONT_RESULT_HOME" \
		/mnt/scripts/$SCRIPT.sh
fi
