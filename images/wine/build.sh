#!/bin/bash

NLP_HOME=$PWD

NLP_WINE=$NLP_HOME/images/wine

IMAGE=nlp-wine

if [ -z "$(docker images | grep $IMAGE)" ]
then
	docker build -t $IMAGE $NLP_WINE
fi

read -p "Введіть шлях до папки з даними: " NLP_DATA

read -p "Введіть шлях до папки, в яку ви хотіли б зберігати дані: " NLP_RESULTS

echo "створюємо скрипт $NLP_WINE/run.sh"

sed -e "s@%NLP_HOME%@$NLP_HOME@g" \
    -e "s@%NLP_DATA%@$NLP_DATA@g" \
    -e "s@%NLP_RESULTS%@$NLP_RESULTS@g" \
    $NLP_WINE/run.template > $NLP_WINE/run.sh
