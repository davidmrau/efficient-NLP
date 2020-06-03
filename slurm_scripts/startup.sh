#!/bin/bash
chmod u+x ../telegram.sh
# download data

DOWNLOAD_FOLDER=../data/
PREFIX=https://rijsbergen.hum.uva.nl/david/
DOWNLOAD_FILE=download_file_list

while read file;do

	wget ${PREFIX}${file} -P $DOWNLOAD_FOLDER 
done < $DOWNLOAD_FILE
pip3 install -r ../requirements.txt --user
