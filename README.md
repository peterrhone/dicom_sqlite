# Training and Using a U-Net Model for Pelvic Organ Detection in CBCTs

## Overview
This project provides a pipeline for training and applying a U-Net model to detect pelvic organs (e.g., prostate, bladder, rectum) in Cone Beam Computed Tomography (CBCT) images. It is designed to assist in treatment planning and dose delivery for radiation oncology, particularly for prostate cancer treatment.

## Requirements

### Hardware
- **GPU**: Minimum 16 GB of RAM
  - Recommended: NVIDIA RTX 3000 Series or newer, or AMD RX 7000 Series or newer with PyTorch support

### Software
- **Python**: Version 3.10 or higher
- **PyTorch**: Required for model training and inference
- **Additional Modules**: See `requirements.txt` for a complete list
- **Operating System**: Linux is recommended due to GTK elements in the GUI. Windows compatibility is untested and may require additional configuration.

## Workflow
> **Note**: If you are only interested in inference with pretrained models, you can skip to **Step 5**. Pretrained models are included in the repository (see `models/` directory). However, these models expect CBCT inputs of size 512x512x88. If your CBCT dimensions differ, you must adjust the model or retrain it (see Step 4).

### 1. Data Preparation
- **Acquire Data**: Collect sufficient CBCT data for training.
- **Data Structure**:
  - Create a data directory (e.g., `/data/`).
  - Place one zip file per patient in the data directory (e.g., `P001.zip`, `P002.zip`, etc.).
  - Each zip file should contain a subdirectory named after the patient (e.g., `P001/`) with hundreds of DICOM files, including Cone Beam CT images and their corresponding contours (e.g., CT and RTSTRUCT files).
  - Expected image dimensions: 512x512x88. If dimensions differ, adjust parameters in the training notebooks.
- **Example Zip Structure** (e.g., `P001.zip`):
  ```
  P001/
      CT1.2.752.XXXXXX.dcm
      CT1.2.752.XXXXXX.dcm
      REG1.XXXXXX.dcm
      RS1.XXXXXX.dcm
      ...
  ```

### 2. Data Processing with `sqlite_dicom2.py`
- **Purpose**: Parses all zip files, extracts DICOM series, and creates a SQLite database containing sagittal slices and masks for the selected organ (e.g., prostate).
- **Output**: A SQLite database file (e.g., `images_prostate.sqlite`) used for training.
- **Usage**:
  - Run `sqlite_dicom2.py` and select the data directory.
  - Choose the organ (e.g., "Prostate") and "Save" mode.
  - The script processes all zip files and stores sagittal slices and masks in the database.

### 3. Inspect Data with `sqlite_viewer.py`
- **Purpose**: Allows browsing and verification of images and masks stored in the SQLite database, which serve as inputs for training.
- **Features**:
  - View CBCT images and corresponding masks.
  - Delete specific images from the database.
  - Toggle mask overlay to verify contour accuracy.
- **Controls**:
  - **Mouse Wheel or Arrow Keys**: Scroll through images.
  - **D Key**: Delete the current image from the database. Careful! No confirmation!
  - **M Key**: Toggle mask overlay (red transparent overlay).
- **Usage**:
  - Run `sqlite_viewer.py` and select the SQLite database (e.g., `images_prostate.sqlite`).
  - Use the interface to inspect and manage data.

### 4. Model Training
- **Location**: Jupyter notebooks (see project directory for specific notebooks).
- **Process**: Training scripts for specific organs (e.g., prostate, bladder, rectum) are documented in the notebooks.
- **Requirements**: Ensure the SQLite database from Step 2 is available and contains sufficient data.

### 5. Model Application and Testing with `demo.py`
- **Purpose**: Apply the trained U-Net model to new data (inference) and visually inspect its performance.
- **Important**: Use patient data from the *validation set* (as listed in the Jupyter notebooks) to ensure a realistic evaluation of the model’s capabilities. Using training data may overestimate performance.
- **Usage**:
  - Run `demo.py` and select patient data from the validation set.
  - Inspect the model’s predictions visually to assess detection accuracy.

## Notes
- Ensure all dependencies in `requirements.txt` are installed before running scripts.
- If processing data with non-standard dimensions, modify parameters in the training notebooks accordingly.
- For issues with the GUI on non-Linux systems, consider running in a Linux environment or virtual machine.
- Pretrained models are located in the `models/` directory. Verify CBCT dimensions match the expected 512x512x88 before using them.