### Slide Extraction

Extract slides out of videos

Note: Only the slide detection bit of the pipeline exists as of now.

### Setup env

```sh
# Download and install virtualenv
virtualenv venv
pip install -r requirements.txt
source venv/bin/activate
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
```
```sh
python model_eval.py
```

### Architecture & memory constraints

TODO


### Workflow to add more data

TODO
