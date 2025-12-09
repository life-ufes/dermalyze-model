# Towards a clinically integrated artificial intelligence tool for triage of skin cancer

This project contains the code for training and evaluating a deep learning model designed for triage in the PAD-UFES-20+ dataset. It utilizes the `raug` library for model training and evaluation.

## Project Structure

- **`benchmarks/`**: Contains scripts for running experiments.
  - `kfold.py`: Main script for running k-fold cross-validation experiments.
  - `train.py`: Contains the training and evaluation logic.
  - `pad20plus/`: Dataset-specific configurations and preprocessing scripts.
- **`models/`**: Contains model definitions and the model hub.
  - `models_hub.py`: Factory for creating model instances.
  - `mobilenet.py`: Implementation of MobileNet models.
  - `metablock.py`: Implementation of the MetaBlock for feature fusion.
- **`config.py`**: Configuration file for dataset paths.
- **`raug/`**: Submodule containing the training and evaluation framework.

## Setup

1.  **Clone the repository and submodules:**
  ```bash
  # Initialize the raug submodule
  git submodule update --init
  ```

2.  **Install dependencies:**
    Ensure you have the required Python packages installed. You can install them using pip:
    ```bash
    python -m pip install -r requirements.txt
    ```

3.  **Configure Dataset Paths:**
    Open `config.py` and update the paths to point to your local copy of the PAD-UFES-20+ dataset.

    ```python
    # config.py
    from pathlib import Path

    DATA_PATH = Path('data')
    PAD_20_PLUS_PATH = Path("/path/to/your/pad-ufes-20-plus") # Update this line
    ```

## Usage

To run the k-fold cross-validation experiments, use the `benchmarks/kfold.py` script. The validation metrics are saved in `benchmarks/pad20plus/results/`.

```bash
python -m benchmarks.kfold
```

## Supported Models

The project currently supports the model MobileNet-V3 (defined in `models/models_hub.py`).
It also supports feature fusion using **MetaBlock** to combine image features with clinical metadata.
