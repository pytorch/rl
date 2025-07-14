#!/usr/bin/env bash

set -e
set -v

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# Create data directory structure
mkdir -p data/scene_datasets
mkdir -p data/datasets

# Set environment variables for Habitat data paths
export HABITAT_DATA_PATH="$(pwd)/data"

echo "Downloading Habitat test scenes..."
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/

echo "Downloading ReplicaCAD scenes..."
python -m habitat_sim.utils.datasets_download --uids replica_cad --data-path data/

echo "Downloading rearrange pick dataset..."
cd data/datasets
wget -O rearrange_pick_replica_cad_v0.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0.zip
unzip -o rearrange_pick_replica_cad_v0.zip
rm rearrange_pick_replica_cad_v0.zip
cd ../..

echo "Downloading point-goal navigation episodes for test scenes..."
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/

echo "Datasets downloaded successfully!"
ls -la data/ 