fileId=1CYIu5InPucyEAr_77xVI8-xNGFdy4Vic
fileName=data.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 


# download data
tar -xzf data.tar.gz
mv data ../
rm data.tar.gz

pip3 install -r ../requirements.txt --user
