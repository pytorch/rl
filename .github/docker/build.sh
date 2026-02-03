#!/bin/bash
# Helper script for building Docker images locally

set -e

REGISTRY="ghcr.io/pytorch"
IMAGE_NAME="torchrl-ci"

function usage() {
    echo "Usage: $0 [base|nightly|stable|habitat] [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cuda-version    CUDA version (e.g., 12.4.0, 11.8.0)"
    echo "  --python-version  Python version (e.g., 3.11, 3.9)"
    echo "  --push            Push to registry after building"
    echo "  --no-cache        Build without cache"
    echo ""
    echo "Examples:"
    echo "  $0 base --cuda-version 12.4.0 --python-version 3.11"
    echo "  $0 nightly --cuda-version 12.4.0 --python-version 3.11 --push"
    exit 1
}

IMAGE_TYPE=${1:-base}
shift || usage

# Default values
CUDA_VERSION="12.4.0"
PYTHON_VERSION="3.11"
PUSH=false
NO_CACHE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Derive short CUDA version
CUDA_SHORT="cuda${CUDA_VERSION%.*}"
CUDA_SHORT="${CUDA_SHORT/.}"

# Derive CU_VERSION for PyTorch
if [[ $CUDA_VERSION == "11.8"* ]]; then
    CU_VERSION="cu118"
elif [[ $CUDA_VERSION == "12.1"* ]]; then
    CU_VERSION="cu121"
elif [[ $CUDA_VERSION == "12.4"* ]]; then
    CU_VERSION="cu124"
else
    CU_VERSION="cpu"
fi

# Build based on type
case $IMAGE_TYPE in
    base)
        TAG="${REGISTRY}/${IMAGE_NAME}:base-${CUDA_SHORT}-py${PYTHON_VERSION}"
        echo "Building base image: $TAG"
        docker build \
            $NO_CACHE \
            -f Dockerfile.base \
            -t "$TAG" \
            --build-arg CUDA_VERSION="$CUDA_VERSION" \
            --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
            .
        ;;
    
    nightly)
        BASE_TAG="base-${CUDA_SHORT}-py${PYTHON_VERSION}"
        DATE=$(date +'%Y%m%d')
        TAG="${REGISTRY}/${IMAGE_NAME}:nightly-${CUDA_SHORT}-py${PYTHON_VERSION}-${DATE}"
        TAG_LATEST="${REGISTRY}/${IMAGE_NAME}:nightly-${CUDA_SHORT}-py${PYTHON_VERSION}-latest"
        
        echo "Building nightly image: $TAG"
        docker build \
            $NO_CACHE \
            -f Dockerfile.nightly \
            -t "$TAG" \
            -t "$TAG_LATEST" \
            --build-arg CUDA_VERSION="$CUDA_VERSION" \
            --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
            --build-arg CU_VERSION="$CU_VERSION" \
            --build-arg BASE_TAG="$BASE_TAG" \
            --build-arg BUILD_DATE="$DATE" \
            .
        ;;
    
    stable)
        BASE_TAG="base-${CUDA_SHORT}-py${PYTHON_VERSION}"
        TAG="${REGISTRY}/${IMAGE_NAME}:stable-${CUDA_SHORT}-py${PYTHON_VERSION}"
        
        echo "Building stable image: $TAG"
        docker build \
            $NO_CACHE \
            -f Dockerfile.stable \
            -t "$TAG" \
            --build-arg CUDA_VERSION="$CUDA_VERSION" \
            --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
            --build-arg CU_VERSION="$CU_VERSION" \
            --build-arg BASE_TAG="$BASE_TAG" \
            .
        ;;
    
    habitat)
        NIGHTLY_TAG="nightly-${CUDA_SHORT}-py${PYTHON_VERSION}-latest"
        TAG="${REGISTRY}/${IMAGE_NAME}:habitat-${CUDA_SHORT}-py${PYTHON_VERSION}"
        
        echo "Building habitat image: $TAG"
        docker build \
            $NO_CACHE \
            -f Dockerfile.habitat \
            -t "$TAG" \
            --build-arg CUDA_VERSION="$CUDA_VERSION" \
            --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
            --build-arg BASE_TAG="$NIGHTLY_TAG" \
            .
        ;;
    
    *)
        echo "Unknown image type: $IMAGE_TYPE"
        usage
        ;;
esac

echo "Build complete: $TAG"

if [ "$PUSH" = true ]; then
    echo "Pushing to registry..."
    docker push "$TAG"
    if [ -n "$TAG_LATEST" ]; then
        docker push "$TAG_LATEST"
    fi
    echo "Push complete"
fi

