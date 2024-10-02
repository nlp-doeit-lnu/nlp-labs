#!/bin/bash

function check_host_os () {

	case "$(uname -s)" in
		"Linux")
			HOST_OS="Linux" ;;

		"Darwin")
			HOST_OS="macOS" ;;

	esac

}

function check_docker_is_active () {

	case $HOST_OS in
		"Linux")
			linux_check_docker ;;

		"macOS")
			macos_check_docker ;;

	esac
	
}

function linux_check_docker () {

	if ! [[ "$(systemctl is-active docker)" == active ]]
	then
		echo "docker daemon is not active."
		echo "we will try to start it automatically."
		echo "or you can do it manually, e.g. using command \" sudo systemctl start docker\"."

		sudo systemctl start docker

	fi

}

function macos_check_docker () {

	if (! docker-stats --no-stream )
	then
		echo "we need to launch Docker."
		open -a Docker
	fi

}

function check_display_var () {
	echo "check DISPLAY variable on host computer."
	case $HOST_OS in
		"Linux")
			DISPLAY_HOST="\$DISPLAY" ;;

		"Darwin")
			DISPLAY_HOST="host.docker.internal:0" ;;

	esac
}

function build_image () {

	echo "check if nlp-java:latest exists"
	if [ -z "$(docker images | grep $IMAGE:$TAG)" ];
	then
		echo "it doesn't exits."
		echo "run docker build."
		
		docker build -t $IMAGE:$TAG $NLP_JAVA

		echo "nlp-java:latest created."
	else
		echo "image nlp-java:latest already exists."
	fi
}

function check_xquartz () {

	echo "Your system is $(uname -s)"
	if [[ "$(uname -s)" = "Darwin" ]] && ! [ -f /opt/X11/bin/xquartz ]
	then
		echo "xquartz is not found, but it is necessary for X11 forwarding."

		install_xquartz
	fi
}

function install_xquartz() {
	
	local answer

        # Prompt to allow install
        echo "xquartz will be installed using Homebrew package manager."

	read -r -p "Proceed with installation? [y - yes / n - no] " answer

	# Default is No
	[ -z "${answer}" ] && answer="n"

	case $answer in
		[Yy]|[Yy][Ee][Ss])
			brew install --cask xquartz	
			;;
		[Nn]|[Nn][Oo])
			echo "unable to start container without X11 forwarding. please install xquartz manually if you want to use this container furthermore."
			exit 0
			;;
		*)
			echo "incomprehensible response. please use y or n"
			;;
		esac
}

# перевіряємо чи існує команда xhost
function check_xhost () {

	echo "checking xhost command ..."
	if ! [ -x "$(command -v xhost)" ]
	then
		echo "command xhost doesn't exist on your system."
		echo "it must be installed."

		install_xhost
	fi
}

function install_xhost () {

	local answer
	read -r -p "install xhost automatically using your package manager? [y - yes | n - no]"

	[ -z "$answer" ] && answer="n"

	case "${answer}" in
		[Yy]|[Yy][Ee][Ss])
			if [ -x "$(command -v apt)" ]
			then
				sudo apt && install xhost
			elif [ -x "$(command -v pacman)" ]
			then
				sudo pacman -S xhost
			elif [ -x "$(command -v yum)" ]
			then
				sudo yum install xhost
			elif [ -x "$(command -v brew)" ]
			then
				brew install --cask xhost
			else
				echo "FAILURE to install xhost. package manager is nowhere to be found."
			fi

			exit 1
			;;
		[Nn]|[Nn][Oo])
			echo "container could not be run without X11-forwarding."
			exit 0
			;;
		*)
			echo "$answer: uncomprehensible answer to the question \"yes or \"."
			echo "please, use y (meaning yes) or n (meaning no)."
			;;
	esac
}

# doesn't work properly yet, just an idea from docker-wine
function add_x11_key () {

    # Check for .Xauthority which is required for authenticating as the current user on the host's X11 server
    if [ -z "${XAUTHORITY}" ]; then
        echo "ERROR: No valid .Xauthority file found for X11"
        exit 1
    fi

    # Get the hex key for the display from host user's .Xauthority file and store in ~/.docker-wine.Xkey
    xauth list | head -n1 | awk '{print $3}' > ~/.nlp-docker.Xkey

    # Lock down permissions
    chmod 600 ~/.nlp-docker.Xkey

    XKEY="$HOME/nlp-docker.Xkey:/root/.Xkey:ro"

}

NLP_HOME=$PWD
NLP_JAVA=$NLP_HOME/images/java
IMAGE=nlp-java
TAG=dev

check_host_os
check_docker_is_active
build_image
check_xquartz
add_x11_key
check_xhost
check_display_var

read -p "enter path to the folder which contains data: " NLP_DATA

read -p "enter path to the folder where you want to save results to: " NLP_RESULTS

echo "creating script $NLP_JAVA/run.sh"

sed -e "s@%NLP_HOME%@$NLP_HOME@g" \
    -e "s@%NLP_DATA%@$NLP_DATA@g" \
    -e "s@%NLP_RESULTS%@$NLP_RESULTS@g" \
    -e "s@%DISPLAY%@$DISPLAY_HOST@g" \
    -e "s@%XKEY%@$XKEY@g" \
    $NLP_JAVA/run.template > $NLP_JAVA/run.sh

chmod u+x $NLP_JAVA/run.sh
