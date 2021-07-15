# CPrecalibration_manuscript_SI

This repository is part of the supporting information to the manuscript in preparation:

** Studying and mitigating the effects of data drifts on ML model performance the example of chemical toxicity data **

Andrea Morger, Marina Garcia de Lomana, Ulf Norinder, Fredrik Svensson, Johannes Kirchmair, Miriam Mathea, and Andrea Volkamer

## Table of contents

* [Objective](#objective)
* [Data and Methods](#data-and-methods)
* [Usage](#usage)
* [License](#license)
* [Acknowledgement](#acknowledgement)
* [Citation](#citation)

## Objective
(Back to [Table of contents](#table-of-contents))

[todo] As modelling strategy, the concept of conformal prediction will be further investigated, with special focus on applicability domain definition. 
A strategy to continuously update models to make reliable predictions for novel data (as in time series) is introduced. 

### Folder Structure
(Back to [Table of contents](#table-of-contents))
- data: input data
- results: output data
- scripts: python scripts
- notebooks: Ipython notebooks

## Data and Methods
(Back to [Table of contents](#table-of-contents))

The ChEMBL datasets used in these notebooks were downloaded from the ChEMBL database vNational Center for Advancing Translational Sciences:
https://tripod.nih.gov/tox21/challenge/data.jsp (downloaded 29.1.2019)

* The molecules were standardised as described in the manuscript (*Data and Methods*)
    * Remove duplicates
    * Use [`standardiser`](https://github.com/flatkinson/standardiser) library (discard non-organic compounds, apply structure standardisation rules, neutralise, remove salts)
    * Remove small fragments and remaining mixtures
    * Remove duplicates

## Usage
(Back to [Table of contents](#table-of-contents))

The notebooks can be used to train aggregated conformal predictors on the provided ChEMBL datasets. Following the experiments shown in the manuscript, they can be recalibrated with predefined update sets.
 
The notebook may be adapted to use the code for different datasets. 

### Installation

!!! [todo] - upload environment.yml, check if required libraries are complete

1. Get your local copy of the `CPrecalibration_manuscript_si` repository by:
    * Downloading it as a [Zip archive](https://github.com/volkamerlab/cprecalibration_manuscript_si/archive/master.zip) and unzipping it, or
    * Cloning it to your computer using git

    ```
    git clone https://github.com/volkamerlab/cprecalibration_manuscript_si.git
    ``` 

2. Install the [Anaconda](
https://docs.anaconda.com/anaconda/install/) (large download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lighter) distribution for clean package version management.

3. Use the package manager `conda` to create an environment (called `recalibration_si`) for the notebooks. You can either use the provided `environment.yml` with which to automatically install all required dependencies (3a, recommended) 
or start with an empty environment (3b) and install the required libraries manually (5).

    * a) Create a conda environment including all dependencies with 
`conda env create -f environment.yml`

    * b) If you prefer to build your own environment, start with 
`conda create --name recalibration_si python=3.8`
   

4. Activate the conda environment: `conda activate recalibration_si`

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
    title = {Studying and mitigating the effects of data drifts on ML model performance the example of chemical toxicity data},
    journal = {manuscript in preparation}
}
```


### Copyright

Copyright (c) 2021, volkamerlab