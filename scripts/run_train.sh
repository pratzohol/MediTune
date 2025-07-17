# create new conda environment

# conda create -n ft python==3.10.6
# conda activate ft
# pip install -r requirements.txt

# prepare dataset
python scripts/prepare_dataset.py

# fine-tune model
python src/main.py