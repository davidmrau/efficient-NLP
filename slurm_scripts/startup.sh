#!/bin/bash
chmod u+x telegram.sh
# download data

PREFIX=https://rijsbergen.hum.uva.nl/david/
DOWNLOAD_FILE=slurm_scripts/download_file_list

pip3 install -r requirements.txt --user

while read file;do
	DOWNLOAD_FOLDER=`echo ${file} | sed 's|\(.*/\).*|\1|'`
	if [ ! -f "${file}" ]; then
        	wget ${PREFIX}${file} -P ${DOWNLOAD_FOLDER}
	fi

done < $DOWNLOAD_FILE

