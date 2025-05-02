# SnowDAR
Have you ever been snow-blinded and needed some assistance in finding those darn snow poles? 

Well, worry no more! Our solution achieves a whopping 0.5 mAP50-95. 
What does this mean?
It's time to get your old GoPro and mount it to your head!

## Installation

Clone the repository
```
git clone git@github.com:TrymNOHG/SnowDAR.git
```

Go into the project

```
cd SnowDAR
```

To get started, it is first recommended that you create a python virtual environment. This can be done by running 
```
python -m venv venv
source venv/bin/activate
```
in your terminal.

After that, install the dependencies by running 
```
pip install -r requirements.txt
```

Copy over the dataset. On IDUN, it would look like this:

```
cp -R /cluster/projects/vc/data/ad/open/Poles .

```

Run the preprocess script which removes the top 35% and cuts the remainding image in half at the center of the x-axis. This may take a few minutes.

```
./process_dataset.sh
```

We are now ready to train a model:
```
python train_yolo.py
```

Alternatively, on IDUN, you can train the model using this line, which will grab a GPU node and start training.

```
./run.sh gpu-slurm.job train_yolo.py
```

To test the model on the test set, run:
```
python test_yolo.py
```

This will test the first training run of the rgb model. If you want to specify which run of the model to use (found in runs/detect/), you can specify it with the run number. Similarly, if you want to test the lidar model, you can specify that as well.
```
python test_yolo.py 5 lidar
```

Note that this will test for our modified version of the dataset, which cuts the images in half. To merge the predictions back into one file, you can run:
```
python process/post_process.py
```


## Exploratory analysis
The file `viz/exploratation.py` includes a number of functions used in the development to understand the dataset.

## Repository Structure
The repository has a separate folder for each section of our exploration. Therefore, the data exploration, visualization, and each model will have its own folder.

