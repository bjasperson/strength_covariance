# Cross-scale covariance for material property prediction
Supporting data and python code for "Cross-scale covariance for material property prediction".
Manuscript link: [arxiv](https://arxiv.org/abs/2406.05146)

## Installation
- [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the repo to your machine
- Create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using the attached environment.yml file 
- From project folder, use `pip install -e .`

## Directories
- data: data files. See separate README.md in directory
- experiments: various notebooks used throughout the research project 
- strength_covariance: main modules

## Module Descriptions
- bibfile_create.py: generates the manuscript bibfile to cite all references for the interatomic potential models used
- data_import.py: uses openKimInterface.py to extract property data from OpenKIM
- dft_import.py: analysis using DFT indicator properties
- explore.py: populate pairplot figures, covariance heatmap and correlation matrix
- factor_usage_plot.py: generates heat map for manuscript
- linear_model.py: regression model code. create model, error analysis and plot generation.
- model_selection.py: explore combinations of factors for model (LOOCV)
- openKimInterface.py: functions to facilitate interface with OpenKIM
