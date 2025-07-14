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

# Note: Using manual downloads to avoid git-lfs prune issues with Habitat utility

# Function to download datasets manually (avoiding Habitat utility git-lfs issues)
download_habitat_dataset() {
    local uid=$1
    local description=$2
    
    echo "Downloading $description manually..."
    
    case "$uid" in
        "habitat_test_scenes")
            # Manual download for test scenes
            cd data/scene_datasets
            if [ ! -d "habitat_test_scenes" ]; then
                git clone https://github.com/facebookresearch/habitat-test-scenes.git habitat_test_scenes || {
                    echo "Manual download failed for $description"
                    return 1
                }
            else
                echo "habitat_test_scenes already exists, skipping download"
            fi
            cd ../..
            ;;
        "replica_cad")
            # Manual download for ReplicaCAD
            cd data/scene_datasets
            if [ ! -d "replica_cad" ]; then
                git clone https://github.com/facebookresearch/replica-cad.git replica_cad || {
                    echo "Manual download failed for $description"
                    return 1
                }
            else
                echo "replica_cad already exists, skipping download"
            fi
            cd ../..
            ;;
        "habitat_test_pointnav_dataset")
            # Manual download for pointnav dataset
            cd data/datasets
            if [ ! -d "habitat_test_pointnav_dataset" ]; then
                wget -O habitat_test_pointnav_dataset.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/habitat-test-scenes/v1/habitat-test-scenes-v1.zip || {
                    echo "Manual download failed for $description"
                    return 1
                }
                unzip -o habitat_test_pointnav_dataset.zip
                rm habitat_test_pointnav_dataset.zip
            else
                echo "habitat_test_pointnav_dataset already exists, skipping download"
            fi
            cd ../..
            ;;
        *)
            echo "Unknown dataset UID: $uid"
            return 1
            ;;
    esac
    echo "$description downloaded successfully!"
}

# Download datasets with fallback
download_habitat_dataset "habitat_test_scenes" "Habitat test scenes"
download_habitat_dataset "replica_cad" "ReplicaCAD scenes"

echo "Downloading rearrange pick dataset..."
cd data/datasets
if [ ! -d "rearrange_pick_replica_cad_v0" ]; then
    wget -O rearrange_pick_replica_cad_v0.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0.zip
    unzip -o rearrange_pick_replica_cad_v0.zip
    rm rearrange_pick_replica_cad_v0.zip
else
    echo "rearrange_pick_replica_cad_v0 already exists, skipping download"
fi
cd ../..

download_habitat_dataset "habitat_test_pointnav_dataset" "Point-goal navigation episodes for test scenes"

echo "Datasets downloaded successfully!"

# Final verification
echo "Verifying downloaded datasets..."
echo "Scene datasets:"
ls -la data/scene_datasets/ 2>/dev/null || echo "No scene_datasets directory found"
echo "Task datasets:"
ls -la data/datasets/ 2>/dev/null || echo "No datasets directory found"

# Check for required datasets
required_scenes=0
if [ -d "data/scene_datasets/habitat_test_scenes" ] || [ -d "data/scene_datasets/replica_cad" ]; then
    required_scenes=1
fi

if [ -d "data/datasets/rearrange_pick_replica_cad_v0" ]; then
    required_datasets=1
else
    required_datasets=0
fi

if [ $required_scenes -eq 1 ] && [ $required_datasets -eq 1 ]; then
    echo "All required datasets are present!"
else
    echo "ERROR: Some required datasets are missing!"
    exit 1
fi 