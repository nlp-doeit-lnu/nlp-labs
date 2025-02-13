#!/usr/bin/bash

NLP_HOME="/home/bohdan/git_projects/NLP-labs"
NLP_WINE="$NLP_HOME/images/wine"

NLP_DATA="/home/bohdan/git_projects/NLP-dcr/data"
NLP_RESULTS="/home/bohdan/git_projects/NLP-dcr/results"

LOCAL_BINS="$NLP_HOME/programs/binaries"
LOCAL_SCRIPTS="$NLP_HOME/scripts"

MOUNT_POINT="/mnt"
CONT_BINS="$MOUNT_POINT/binaries"
CONT_DATA="$MOUNT_POINT/data"
CONT_SCRIPTS="$MOUNT_POINT/scripts"
CONT_RESULTS="$MOUNT_POINT/results"




$NLP_WINE/docker-wine --cache \
	--local \
	--as-root \
	--volume="$LOCAL_BINS:$CONT_BINS" \
	--volume="$LOCAL_SCRIPTS:$CONT_SCRIPTS" \
	--volume="$NLP_DATA:$CONT_DATA:ro" \
	--volume="$NLP_RESULTS:$CONT_RESULTS" \
	$CONT_SCRIPTS entry.sh
