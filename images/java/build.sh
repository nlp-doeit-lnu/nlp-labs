#!/bin/bash

NLP_HOME=$PWD

NLP_JAVA=$NLP_HOME/images/java/

docker build -t nlp-java $NLP_JAVA

read -p "Введіть шлях до папки з даними: " NLP_DATA

read -p "Введіть шлях до папки, в яку ви хотіли б зберігати дані: " NLP_RESULTS

cat > $NLP_JAVA/run.sh << EOF
#!/bin/bash

xhost +local:root

# вкажіть власні шляхи до папок
NLP_HOME="$NLP_HOME"

NLP_DATA="$NLP_DATA"
NLP_RESULTS="$NLP_RESULTS"

LOCAL_JARS="$NLP_HOME/programs/jars"
LOCAL_SCRIPTS="$NLP_HOME/scripts"
CONT_JARS="/mnt/jars"
CONT_DATA="/mnt/data"
CONT_SCRIPTS="/mnt/scripts"
CONT_RESULTS="/mnt/results"

LOCAL_X11_PATH="/tmp/.X11-unix"
CONT_X11_PATH="/tmp/.X11-unix"

VIDEOCARD="/dev/dri"
WEB_PORT="5800:5800"
RDP_PORT="5900:5900"

DOCKER_IMAGE="nlp-java:latest"

if [ -z "\$1" ]
then
	echo "не введено назву програми"
else
	SCRIPT=\$1

	docker run --device \$VIDEOCARD \\
		--rm \
		-p \$WEB_PORT \\
		-p \$RDP_PORT \\
		-v \$LOCAL_JARS:\$CONT_JARS \\
		-v \$LOCAL_SCRIPTS:\$CONT_SCRIPTS \\
		-v \$NLP_DATA:\$CONT_DATA \\
		-v \$NLP_RESULTS:\$CONT_RESULTS \\
		-v \$LOCAL_X11_PATH:\$CONT_X11_PATH \\
		-e DISPLAY=\$DISPLAY \\
		\$DOCKER_IMAGE \$CONT_SCRIPTS/\$SCRIPT.sh
fi

xhost -local:root
EOF

chmod u+x $NLP_JAVA/run.sh
