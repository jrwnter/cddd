#!/bin/bash
fileid="1oyknOulq_j0w9kzOKKIHdTLo5HphT99h"
filename="default_model.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip default_model.zip -d cddd/data/
