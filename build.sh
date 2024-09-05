#!/bin/bash

if [[ $1 = "java" ]]
then
	./images/java/build.sh
elif [[ $1 = "wine" ]]
then
	./images/wine/build.sh
else
	echo "ПОМИЛКА: введіть назву зображення."
fi
