# Fire Detection with Himawari-8 and FIRMS – Reproducible Pipeline

This repository contains the core components of the fire-detection pipeline described in the associated manuscript. It covers:

1. Extraction of Himawari-8 AHI input channels for a specified Region of Interest (ROI).
2. Construction of pixel-wise fire targets from FIRMS active-fire products on the AHI grid.
3. Construction of train/validation/test NPZ datasets by pairing AHI inputs with FIRMS-derived targets, including a stratified split by fire activity.
4. Training and evaluation of U-Net / Attention U-Net models for active-fire detection, including custom Keras callbacks for checkpointing and logging.

The aim is to make the end-to-end workflow reproducible and extensible to other regions and time periods.

---

## Repository structure

- `createDataInputHimawari.py`  
  Script to **extract Himawari-8 AHI channels** from Level-1B NetCDF files over a given Region of Interest (ROI). It:
  - scans a directory of `.nc` files,
  - reads latitude and longitude from each file,
  - crops the selected channel(s) (e.g. `tbb_07`) to the specified lat/lon bounds,
  - writes per-timestamp NPZ files (e.g. `YYYYMMDD_HHMM.npz`) into channel-specific folders such as `ch7/`, with the cropped array stored under the key `channel_data`.

- `createDataTarget.py`  
  Script to construct **pixel-wise fire masks (targets)** on the AHI grid from FIRMS active-fire detections. It aggregates FIRMS hotspot points per timestamp, rasterises them onto the Himawari-8 grid (using the same ROI and resolution as the inputs), applies the chosen buffering/dilation, and saves per-timestamp NPZ target files (e.g. `YYYYMMDD_HHMM.npz`) with a 2D fire mask (e.g. under key `matrix`).

- `createDataSet.py`  
  Script to build **train/validation/test NPZ datasets** by pairing AHI input NPZ files with the FIRMS-derived target NPZ files. It:
  - checks that each selected timestamp has both complete input channels and a target,
  - counts fire pixels per timestamp,
  - builds quantile-based bins over fire counts,
  - performs a **stratified temporal split** (optionally grouping by date so that a single date does not span multiple splits),
  - loads and stacks the requested AHI channels into `(H, W, C)` arrays,
  - saves final `train/val/test` NPZ files containing `input_np`, `target_np`, and `file_names`.

- `fire-detection.ipynb`  
  Jupyter notebook that implements the **model training and evaluation**:
  - defines the U-Net / Attention U-Net architecture(s),
  - loads the NPZ datasets produced by `createDataSet.py`,
  - configures the optimizer, loss, and metrics,
  - implements and uses custom Keras callbacks such as `CustomModelCheckpoint` and `CustomHistory`,
  - trains the model and computes the pixel-wise and object-wise metrics reported in the manuscript.

You may also include:

- `README.md` (this file),
- `requirements.txt` (Python dependencies),
- optionally a `callbacks.py` module if you decide to move the custom callbacks out of the notebook.

---

## Data sources

The pipeline relies on publicly available remote-sensing products:

- **Himawari-8 AHI**  
  Geostationary satellite imagery (e.g. Band 7 thermal infrared) from JMA/JAXA.  
  Access: refer to the Himawari-8 Level-1B / gridded products distribution (e.g. via JAXA or affiliated data portals).

- **NASA FIRMS**  
  MODIS/VIIRS active-fire detections from the Fire Information for Resource Management System (FIRMS).  
  Access: NASA EOSDIS FIRMS.

Due to data volume and licensing constraints, this repository does **not** redistribute the raw satellite products. Instead, it provides the scripts necessary to:

1. Convert Himawari-8 Level-1B NetCDF files into cropped NPZ channel stacks over a specified ROI (`createDataInputHimawari.py`).
2. Convert FIRMS point detections into gridded AHI-aligned fire masks (`createDataTarget.py`).
3. Combine the preprocessed AHI channels and the gridded targets into NPZ datasets for modelling (`createDataSet.py`).

Users are expected to download the raw data from the original providers and adapt the file paths in the scripts to match their local directory structure.

---

## Workflow

A typical end-to-end workflow is:

1. **Prepare raw data**
   - Download Himawari-8 AHI files for the study region and period.
   - Download FIRMS MODIS/VIIRS active-fire detections for the same period.

2. **Extract AHI input channels (NetCDF → NPZ)**  
   Edit the configuration inside `createDataInputHimawari.py`:

   ```python
   data_dir   = r".../path/to/himawari/netcdf"
   channels   = ["tbb_07"]          # list of channels to extract
   root_dir   = r".../path/to/ahi_npz_root"
   lat_lower, lat_upper = ...
   lon_lower, lon_upper = ...
   ```

   Then run:

   ```bash
   python createDataInputHimawari.py
   ```

   This will create channel-specific folders (e.g. `ch7/`) under `root_dir`, each containing cropped NPZ files per timestamp.

3. **Construct pixel-wise targets (FIRMS → AHI grid)**  
   Edit the input/output paths inside `createDataTarget.py` to point to your raw FIRMS and AHI data, then run:

   ```bash
   python createDataTarget.py
   ```

   This should produce per-timestamp NPZ target files (e.g. `YYYYMMDD_HHMM.npz`) containing a 2D fire mask aligned with the AHI grid (e.g. under key `matrix`).

4. **Build train/validation/test NPZ datasets**  
   Edit the configuration block at the bottom of `createDataSet.py`:

   ```python
   if __name__ == "__main__":
       input_dir       = r".../path/to/ahi_npz_root"
       target_dir      = r".../path/to/target_npz"
       channel_indices = [7]              # e.g. [7] for Band 7 only, or [7, 13] etc.
       input_key       = "channel_data"
       target_key      = "matrix"
       threshold       = None             # None for binary masks; or a numeric threshold
       n_bins          = 5
       ratios          = (0.70, 0.15, 0.15)
       seed            = 42
       group_by_date   = True

       dataset_name    = "datasetIR(7)_buffer10m_Nofilter"
       save_dir        = r".../path/to/output_npz"
       ...
   ```

   Then run:

   ```bash
   python createDataSet.py
   ```

   This will:
   - ensure matched input/target pairs,
   - compute fire-pixel counts per timestamp,
   - perform a stratified split into train/val/test,
   - save final NPZ files such as:
     - `datasetIR(7)_buffer10m_Nofilter_train.npz`
     - `datasetIR(7)_buffer10m_Nofilter_val.npz`
     - `datasetIR(7)_buffer10m_Nofilter_test.npz`

5. **Train and evaluate the model**  
   Open `fire-detection.ipynb` in Jupyter:

   ```bash
   jupyter notebook fire-detection.ipynb
   ```

   In the notebook, configure:

   - paths to the NPZ datasets created by `createDataSet.py`,
   - model hyperparameters (e.g. number of filters, depth, learning rate),
   - training configuration (epochs, batch size),
   - callbacks such as `CustomModelCheckpoint` and `CustomHistory`.

   Run all cells to:
   - train the U-Net / Attention U-Net,
   - compute pixel-wise ROC–AUC and PR–AUC (with bootstrap confidence intervals, if implemented),
   - compute object-wise metrics using your chosen matching criteria,
   - generate the figures and tables corresponding to the manuscript.

---

## Environment and dependencies

The code has been developed and tested with Python 3.x and uses standard scientific Python libraries (e.g. NumPy, TensorFlow/Keras, scikit-learn, Matplotlib, tqdm). To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

You may need to adjust version constraints or add additional packages depending on your local setup and exact use of the scripts.

---

## Reproducibility and DOI

For reproducibility, this repository is intended to be archived with a persistent DOI (e.g. via Zenodo). The archived snapshot will include:

- the dataset-construction scripts (`createDataInputHimawari.py`, `createDataTarget.py`, `createDataSet.py`),
- the training and evaluation notebook (`fire-detection.ipynb`),
- this `README.md`,
- the `requirements.txt` file.

In the manuscript, please refer to this repository and its DOI as the primary source for reproducing the pipeline and extending it to other regions or periods.

---

## Contact and citation

If you use this code or pipeline in your research, please cite the software release and, once available, the associated article.

**Software citation**

Patombongi, A. (2025). *fire-detection-himawari-firms: Fire detection pipeline for Himawari-8 and FIRMS* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17873839

BibTeX:

```bibtex
@software{patombongi_fire_detection_himawari_firms_2025,
  author    = {Patombongi, Andi},
  title     = {fire-detection-himawari-firms: Fire detection pipeline for Himawari-8 and FIRMS},
  year      = {2025},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.17873839},  
  url       = {https://doi.org/10.5281/zenodo.17873839}
}

For questions or issues, feel free to open an issue on the repository or contact the corresponding author.
