
# download data
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=13FlxE2GN9pMAc7K0-iT8YWqVGkGUZZ43' -O data.tar.gz
tar -xzf data.tar.gz -C ../
rm data.tar.gz

pip3 install -r ../requirements.txt --user
