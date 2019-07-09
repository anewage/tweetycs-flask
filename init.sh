#!/bin/bash

echo "Downloading Gdown..."
wget https://github.com/circulosmeos/gdown.pl/archive/master.zip .
unzip master.zip
echo "Gdown downloaded!"

cd gdown.pl-master
echo "Downloading ML models..."
./gdown.pl https://drive.google.com/open?id=11hj3rv6bYhn4tMZQBb0BmO5KCqVN90RG models.zip
echo "Download complete!"

echo "Moving models..."
mv models.zip ../models/
echo "Models moved!"

cd ../models/
echo "Unzipping models..."
unzip models.zip

echo "Done unzipping!"
echo "Starting the flask application"
python mlapp.py; python dlapp.py
