#!/bin/bash 


if [ -n "$VIRTUAL_ENV" ]; then
  	echo "A virtual environment is already activated: $VIRTUAL_ENV, use deactivate command to exit "
else
  	echo "No virtual environment is currently activated."
  	# enter in the virtual environment of the project
	. ./env/bin/activate
  	if [ $? -eq 0 ]; then
  		echo "Virtual environment activated, use deactivate to exit"
  	else
  		echo "It's not possible to activate the virtual environment, please check the folder position"

	fi
fi




