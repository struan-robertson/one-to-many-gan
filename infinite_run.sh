#!/bin/bash

for ((i=1;;i++)); do
	printf '\n\n============== Run #%s ==============\n\n' "$i"
	python train.py
done
