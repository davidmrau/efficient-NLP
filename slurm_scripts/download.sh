PREFIX=https://rijsbergen.hum.uva.nl/david/

while read file;do
	DOWNLOAD_FOLDER=`echo ${file} | sed 's|\(.*/\).*|\1|'`
	if [ ! -f "${file}" ]; then
        	wget ${PREFIX}${file} -P ${DOWNLOAD_FOLDER}
	fi

done < $1
