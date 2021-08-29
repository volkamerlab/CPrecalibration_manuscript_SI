# CPrecalibration_manuscript_SI

This repository is part of the supporting information to the manuscript in preparation:

**Studying and mitigating the effects of data drifts on ML model performance at the example of chemical toxicity data**

Andrea Morger, Marina Garcia de Lomana, Ulf Norinder, Fredrik Svensson, Johannes Kirchmair, Miriam Mathea, and Andrea Volkamer

## Table of contents

* [Objective](#objective)
* [Data and Methods](#data-and-methods)
* [Usage](#usage)
* [License](#license)
* [Citation](#citation)

## Objective
(Back to [Table of contents](#table-of-contents))

In this work, a strategy to recalibrate conformal prediction models to mitigate the effects of data drifts was presented. 
The strategy was analysed with respect to temporal data drifts in time-split ChEMBL datasets as well as observed drifts between internal and external data. 

This repository focuses on the ChEMBL data and illustrates how the recalibration strategy is applied at the example of 12 collected ChEMBL datasets, extracted from ChEMBL Version 26.

* Notebook `1_continuous_calibration_example.ipynb` explains the recalibration concept and application at the example of endpoint `ChEMBL228`. The notebook can easily be adapted to be used for the other provided ChEMBL data sets or your own data.
* Notebook `2_continuous_calibration_evaluate_multiple_endpoints.ipynb` shows how the recalibration experiments can be performed at once for all 12 ChEMBL endpoints as used in the manuscript. The notebook can also be customised to build CP models and make predictions for your own data.

* Dataset `CHEMBL228_chembio_normalizedDesc.csv.tar.bz2` contains the input data for the `ChEMBL228` calculations. The file contains the molecule ChEMBL IDs, SMILES, binary activity, publication year, and ChemBio descriptors. 
Note that due to the size of the datasets with the ChemBio descriptors, only one dataset is provided with this GitHub repo. This is sufficient to run the `continuous_calibration_example.ipynb` notebook. To be able to run the full pipeline, please download the compressed file with data for all 12 endpoints used in this work from Zenodo under this [link](link_to_zenodo) and copy it to the data folder.
* Dataset `data_size_chembio_chembl.csv` holds the precalculated information which year is used per data set to create the time split data while retaining specific ratios. It contains the information for all 12 endpoints, wich will be connected to the data in the notebooks. See the manuscript for details on the data splitting.

## Data and Methods
(Back to [Table of contents](#table-of-contents))

The ChEMBL datasets used in these notebooks were downloaded from the ChEMBL database version 26.

* The molecules were standardised as described in the manuscript (*Data and Methods*) in a KNIME workflow including the following steps
    * Remove solvents and salts
    * Annotate aromaticity
    * Neutralise charges
    * Mesomerise structures 
    * Remove duplicates
* For each molecule the following information is stored:
    * molecule chembl ID
    * SMILES
    * binary activity (i.e. 1 if measured active, 0 if measured inactive in the respective assay)
    * publication year
    * ChemBio descriptors (calculated as described in [Garcia de Lomana et al., JCIM, 2021, 61, 7, 3255–3272](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00451)).

Recalibration of conformal prediction models provides a strategy to mitigate effects of data drifts between training and test data. Instead of updating and retrainining a conformal prediction model, it suggests to only update the calibration set with new data, that are closer to the test set (than the original calibration data). For more information on conformal prediction and the recalibration strategy, we refer to the manuscript or to our previous work ([Morger et al., JCheminf, 2021, 13, 35](https://link.springer.com/article/10.1186/s13321-021-00511-5)).

## Usage
(Back to [Table of contents](#table-of-contents))

The notebooks can be used to train aggregated conformal predictors on the provided ChEMBL datasets. Following the experiments shown in the manuscript, they can be recalibrated with predefined update sets.
 
The notebook may be adapted to use the code for different datasets. 

### Installation

1. Get your local copy of the `CPrecalibration_manuscript_si` repository by:
    * Cloning it to your computer using git

    ```
    git clone https://github.com/volkamerlab/cprecalibration_manuscript_si.git
    ``` 

2. Install the [Anaconda](
https://docs.anaconda.com/anaconda/install/) (large download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lighter) distribution for clean package version management.

3. Use the package manager `conda` to create an environment (called `recalibration_si`) for the notebooks. You can either use the provided `environment.yml` with which you can automatically install all required dependencies (3a, recommended) 
or start with an empty environment (3b) and install the required libraries manually (5).

    * a) Create a conda environment including all dependencies with 
`conda env create -f environment.yml`

    * b) If you prefer to build your own environment, start with 
`conda create --name recalibration_si python=3.8`
   

4. Activate the conda environment: `conda activate recalibration_si`

If you created your conda environment from the `environment.yml` file provided with this repository, it is now ready to be used with the notebooks. If you are building your own environment, continue with step 5 to install the required packages.

5. Install packages (only required together with 3b): 
    * If you successfully created your environment from the `environment.yml` file (3a), this step 5 can be skipped. 
    * If you started with your own environment (3b), continue by installing the following libraries: 
   
    
        `conda install pandas`
    
        `conda install matplotlib`
    
        `conda install -c conda-forge scikit-learn=0.22.2`
    
        `pip install https://github.com/morgeral/nonconformist/archive/master.zip`
        
        `conda install -c conda-forge notebook`
        
        `conda install -c conda-forge umap-learn`
        

## License
(Back to [Table of contents](#table-of-contents))

This work is licensed under the MIT License.

## Citation
(Back to [Table of contents](#table-of-contents))

If you make use of the `CPrecalibration_manuscript_SI` notebook, please cite:

```
@article{cprecalibration,
    author = {
        Morger Andrea, 
        Garcia de Lomana Marina,
        Norinder Ulf,
        Svensson Fredrik, 
        Johannes Kirchmair,
        Miriam Mathea,
        Volkamer Andrea},
    title = {Studying and mitigating the effects of data drifts on ML model performance at the example of chemical toxicity data},
    journal = {manuscript in preparation}
}
```


### Copyright

Copyright (c) 2021, volkamerlab
