<<<<<<< Updated upstream
#!/bin/bash
chmod u+x ../telegram.sh
# download data
=======
PREFIX=https://rijsbergen.hum.uva.nl/david/
DOWNLOAD_FILE=download_file_list

chmod u+x ../telegram.sh
>>>>>>> Stashed changes

DOWNLOAD_FOLDER=../data/
PREFIX=https://rijsbergen.hum.uva.nl/david/
DOWNLOAD_FILE=download_file_list

while read file;do

	wget ${PREFIX}${file} -P $DOWNLOAD_FOLDER 
done < $DOWNLOAD_FILE
pip3 install -r ../requirements.txt --user

while read file;do
	DOWNLOAD_FOLDER=`echo ${file} | sed 's|\(.*/\).*|\1|'`
        wget ${PREFIX}${file} -P ../${DOWNLOAD_FOLDER}
done < $DOWNLOAD_FILE

