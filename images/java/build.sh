#!/bin/bash

NLP_HOME=$PWD

NLP_JAVA=$NLP_HOME/images/java

IMAGE=nlp-java

if [ -z "$(docker images | grep $IMAGE)" ]
then
	docker build -t $IMAGE $NLP_JAVA
fi

if [ -z "$(which xhost)" ]
then
	echo "потрібно встановити xhost"
	sudo apt update
	sudo apt install
fi

read -p "Введіть шлях до папки з даними: " NLP_DATA

read -p "Введіть шлях до папки, в яку ви хотіли б зберігати дані: " NLP_RESULTS

echo "створюємо скрипт $NLP_JAVA/run.sh"

sed -e "s@%NLP_HOME%@$NLP_HOME@g" \
    -e "s@%NLP_DATA%@$NLP_DATA@g" \
    -e "s@%NLP_RESULTS%@$NLP_RESULTS@g" \
    $NLP_JAVA/run.template > $NLP_JAVA/run.sh
