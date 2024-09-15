#!/bin/bash

if [[ $1 = "java" ]]
then
	sh images/java/build.sh
elif [[ $1 = "wine" ]]
then
	sh images/wine/build.sh
else
	echo "ПОМИЛКА: введіть назву зображення."
fi
