# Exploring covariance between predicted strength and canonical (predictor) properties

## Installation
- [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the repo to your machine
- Create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using the attached environment.yml file 
- From project folder, use `pip install -e .`

## Module Descriptions
- data_import.py: uses openKimInterface.py to extract property data from OpenKIM
- explore.py: populate pairplot figures, covariance heatmap and correlation matrix
- model_selection.py: explore combinations of factors for model (LOOCV and RepeatedKFold CV)
- uncertainty_quantification.py: perform bootstrap uncert quantification on model
