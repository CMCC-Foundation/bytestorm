# ByteStorm

This is the official repository for the paper entitled: "**ByteStorm: a data-driven, multi-step approach for Tropical Cyclone detection and tracking**", using deep learning and the BYTE tracking algorithm.

## Overview

ByteStorm is a comprehensive framework for detecting and tracking tropical cyclones (TCs) across East- and West- North Pacific (ENP and WNP) basins basins using Machine Learning and Computer Vision techniques. The system combines two Deep Learning models for TC *classification* and *localization* with the BYTE tracking algorithm to generate continuous, accurate TC tracks over time.

## Architecture

ByteStorm follows a **VGG**-based backbone architecture with specialized adaptations for tropical cyclone tracking:

- **Detection Phase**: Spatial localization using VGG networks. The TC Detection is divided into two steps: a classification phase where the input is classified as TC or no-TC, followed by a localization phase where (if present) the TC eye is localized.
- **Tracking Phase**: BYTE-based multi-object tracking with Kalman filtering to align TC tracks over time.

### Input Specifications

The model processes **2D Weather Fields**, in particular: *relative vorticity at 850mb* (RV850) and *mean sea level pressure* (MSLP).

## Project Structure

```
bytestorm/
├── src/                                # Main source code (to launch)
│   ├── training.py                     # Model training pipeline
│   ├── inference.py                    # Inference and detection
│   ├── build_dataset_patches.py        # Dataset building
│   ├── era5_download.py                # Data download
│   ├── scaler.py                       # Data standardization
│   ├── info.py                         # Training utilities
│   ├── train.sh                        # Training launch file
│   └── config/                         # Configuration files for training
│
├── resources/library/tropical_cyclone/ # Main library source code
│   ├── cyclone.py                      # Main TC detection and tracking module
│   ├── dataset.py                      # PyTorch TC Dataset
│   ├── models.py                       # Model architectures
│   ├── era5.py                         # ERA5 download interface
│   ├── scaling.py                      # Feature scalers
│   ├── callbacks.py                    # Training callbacks
│   ├── diskio.py                       # I/O operations
│   ├── utils.py                        # Utility functions
│   ├── sampler.py                      # Data sampling
│   └── _cyclone/                       # Internal cyclone modules
│       ├── augmentation.py             # Data augmentation
│       ├── georef.py                   # Georeference utilities
│       ├── inference.py                # Inference pipeline
│       ├── patch.py                    # Patch extraction
│       ├── visualize.py                # Visualization tools
│       ├── macros.py                   # Constants and macros
│       ├── utils.py                    # Helper functions
│       └── tracker/                    # Tracking implementation
│           ├── byte_tracker.py         # BYTE tracking algorithm
│           ├── kalman_filter.py        # Kalman filter implementation
│           ├── matching.py             # Track matching strategies
│           └── basetrack.py            # Base tracking classes
│
└── notebooks/                          # Jupyter notebooks to plot the paper's figures
    ├── BYTE-tracking-algorithm.ipynb
    ├── comparison_table_*.ipynb
    └── paper-figures-*.ipynb

```

## Installation

### Requirements

- Python 3.8+
- PyTorch/CUDA support
- ERA5 data access (via cdsapi Copernicus Climate Change Service)
- Dynamicopy 0.6.1

### Setup

```bash
# Clone the repository
git clone https://github.com/CMCC-Foundation/bytestorm.git
cd bytestorm

# Install dependencies (adjust based on your environment)
pip install -r requirements.txt

# Configure ERA5 data access
# Follow Copernicus Climate Change Service documentation for credentials
```

## Usage

### Data Preparation

```bash
# Download ERA5 data
python src/era5_download.py

# Build dataset patches
python src/build_dataset_patches.py

# Compute standard scaler
python src/scaler.py
```

### Training

```bash
# Train localization model
mpirun src/training.py --config src/config/config_localization_model.toml --devices GPUS_PER_NODE --num_nodes NUM_NODES

# Train classification model
mpirun src/training.py --config src/config/config_classification_model.toml --devices GPUS_PER_NODE --num_nodes NUM_NODES
```

### Inference and Tracking

```bash
# Run inference pipeline
python src/inference.py LOC_MODEL_NAME CLS_MODEL_NAME CLS_THRESHOLD 

# See notebooks/BYTE-tracking-algorithm.ipynb for detailed examples
```

### Tracking Configuration

The BYTE tracker can be configured with:

```python
from tropical_cyclone.cyclone import BYTETracker

tracker = BYTETracker(
    track_thresh=0.5,  # Detection confidence threshold
    track_buffer=4,    # Maximum frames to keep lost tracks
    match_thresh=0.8,  # IoU threshold for matching
    lats=lats,         # Latitude array
    lons=lons,         # Longitude array
    mot20=False,       # MOT20 evaluation mode
    frame_rate=1,      # Time steps per second (fixed for TC data)
)
```

## Performance Evaluation

The framework includes comprehensive evaluation tools:

- **POD/FAR metrics**: Probability of Detection and False Alarm Rate
- **Track matching**: Spatial and temporal alignment with reference datasets (e.g., IBTrACS)
- **Comparative analysis**: Performance comparison with traditional and ML-based trackers

See notebooks for comparative tables and detailed performance analysis:
- `comparison_table_between_ML_trackers.ipynb`
- `comparison_table_between_traditional_and_ML_trackers.ipynb`

## Citation

If you use ByteStorm in your research, please cite:

```
ByteStorm: A Multi-Step Data-Driven Approach for Tropical Cyclone Detection and Tracking
```

## Contributors

**Advanced Digital Innovation Centre (ADIC)** - CMCC Foundation

- **Davide Donno** ([davide.donno@cmcc.it](mailto:davide.donno@cmcc.it))
- **Marco De Carlo** ([marco.decarlo@cmcc.it](mailto:marco.decarlo@cmcc.it))
- **Donatello Elia** ([donatello.elia@cmcc.it](mailto:donatello.elia@cmcc.it))

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- BYTE tracking algorithm adapted from [ByteTrack](https://github.com/FoundationVision/ByteTrack)
- ERA5 data provided by Copernicus Climate Change Service
- IBTrACS reference data for validation

## References

### Key Publications

- [BYTE: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- [Intercomparison of four algorithms for detecting tropical cyclones using ERA5](https://gmd.copernicus.org/articles/15/6759/2022/)

### Data Sources

- [Copernicus Climate Change Service - ERA5](https://cds.climate.copernicus.eu/)
- [International Best Track Archive for Climate Stewardship (IBTrACS)](https://www.ncei.noaa.gov/products/international-best-track-archive)

## Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the development team.
