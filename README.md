### Slide Extraction

Extract slides out of videos

Note: Only the slide detection bit of the pipeline exists as of now.

### Setup

You need to have ffmpeg installed & in path.
I've noticed some inconsistencies between ffmpeg versions. Please use v3.0

```sh
# Create a new env using virtualenv
virtualenv venv
source venv/bin/activate
```
Install tensorflow from 
https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#pip-installation

```sh
pip install -r requirements.txt
```

### Import data

```sh
# The script downloads a bunch of youtube videos & serialzes them into bin
# files stored at data/serialized/
python import_data.py
```

### Train & evaluate

```sh
python model_train.py

# Resume from checkpoint
python model_train.py --resume_training
# This will surely screw up your event log as you're going back in time to the
# checkpoint, which is saved every 100 steps or so.
# Tensorboard will end up showing some weird graphs. Increment run_no to avoid this
python model_train.py --resume_training --run_no=2
```
```sh
# Continously evaluate the model every 5 minutes
python model_eval.py
# To evaluate pre-trained weights, use this
# You can download my pre-trained weights from
# https://drive.google.com/file/d/0B8gGhyLfZXnYazlaRlJqeURMS2c/view?usp=sharing
# They give me around 97.5% accuracy as of now
python model_eval.py --run_once --checkpoint_dir=./learned_params
```

### Visualize
```sh
# In case you've changed your default logdir
tensorboard --logdir ./tmp/train

# For multiple runs, change the add a run_no flag to training
python model_train.py --run_no=2
# This allows you to compare multiple runs on tensorboard
```

### Serve

TODO

### Architecture & memory constraints

TODO


### Workflow to add more data

TODO
