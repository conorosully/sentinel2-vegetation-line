# sentinel2-vegetation-line
Detecting sub-pixel changes in the coastal vegetation line with Sentinel-2 imagery

This repository contains the code required to reproduce the results in the conference paper:

> [coming soon]

This code is only for academic and research purposes. Please cite the above paper if you intend to use whole/part of the code. 

## Data Files

We have used the following dataset in our analysis: 

1. The Sentinel-2 Irish Vegetation Edge (SIVE) Dataset [here](https://zenodo.org/records/17122999).

 The data is available under the Creative Commons Attribution 4.0 International license.

## Code Files
You can find the following files in the src folder:

- `0_process_rasters.ipynb` Stack Sentinel-2 scenes, Guidance band and vegetation lines so they can be used to create a modelling dataset.
- `0_process_rasters_erosion.ipynb` Process Sentinel-2 scenes so they can be used to calculate erosion rates.
- `1_create_model_dataset.ipynb` Create training and test data for edge detection machine learning models.
- `2_model_evaluation.ipynb` Produce metrics and visualisations for the performance of all edge detection models.
- `3_calculate_average_lines.ipynb` Calculate the average detected vegetation line (AVDLs) and format results for DSAS.
- `4_compare_erosion_rates.ipynb` Visualise and summarise the erosion rates and distance calculations from DSAS.
- `5_figures_for_paper.ipynb` Additional figures for research paper
- `utils.py` Helper functions used to perform the analysis. 
- `evaluation.py` Help functions used to evaluate the edge detection models.
- `network_hed.py` Holistic nested edge detection architecture and backbones.
- `train.py` Training loop and hyperparameter testing.
