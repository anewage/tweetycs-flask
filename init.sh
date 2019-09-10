#!/bin/bash
rm -rf models/README.md
if [ "$(ls -A ./models)" ]; then
	echo "Models are already loaded!"
	echo "Starting the flask application"
else
	echo "Downloading Gdown..."
	wget https://github.com/circulosmeos/gdown.pl/archive/master.zip .
	unzip master.zip
	echo "Gdown downloaded!"

	cd gdown.pl-master
	echo "Downloading ML models..."
	./gdown.pl https://drive.google.com/open?id=11hj3rv6bYhn4tMZQBb0BmO5KCqVN90RG models.zip
	echo "Downloading training data..."
	./gdown.pl https://drive.google.com/open?id=19M69Oz8zvizxj0RzWqfLRIExxUS4Keu8 test_data.zip
	echo "Download complete!"

	echo "Moving models..."
	mv models.zip ../models/
	echo "Models moved!"
	echo "Moving Data..."
	mv test_data.zip ../data/
	echo "Moving complete!"

	cd ../models/
	echo "Unzipping models..."
	unzip models.zip
	cd ../data/
	echo "Unzipping data..."
	unzip test_data.zip
	echo "Done unzipping!"
	cd ../

	echo "Cleaning up..."
	rm -rf models/models.zip
	rm -rf data/test_data.zip
	rm -rf gdown.pl-master
	rm -rf master.zip
	echo "Starting the flask application"
fi
echo "RUNNING $SCRIPT ..."
python $SCRIPT
