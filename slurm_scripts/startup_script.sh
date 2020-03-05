
# download data
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=FILEID' -O ../data.tar.gz
tar -xzf data.tar.gz -C ../

pip3 install -r ../requirements.txt --user
