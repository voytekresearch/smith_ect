# smith_ect

`smith_ect` project repository: investigating the roles of aperiodic activity and slow oscillations in electroconvulsive therapy (ECT) as a treatment for Major Depressive Disorder (MDD)

[[Paper](https://www.medrxiv.org/content/10.1101/2022.04.15.22273811v1)]

## Overview

ECT is one of the most efficacious interventions for treatment-resistant depression. Despite its efficacy, ECT’s neural mechanism of action remains unknown. Although ECT has been associated with “slowing” in the electroencephalogram (EEG), how this change relates to clinical improvement is unresolved. Until now, increases in slow-frequency power have been assumed to indicate increases in slow oscillations, without considering the contribution of aperiodic activity, a process with a different physiological mechanism. 

Here, we explore the properties of EEG power spectra from 9 patients with MDD recieveing a course of ECT treatment. We use spectral parameterization to disambiguate the contributions of aperiodic activity and the power in delta (1-3 Hz) oscillatory peaks to the EEG "slowing" effect. Furthermore, we invesigate the relationships between these measures and clinical improvement as measured by the Quick Inventory of Depressive Symptomatology (QIDS) scale.


## Project Guide

`00-Preprocessing` contains scripts used to preprocess scalp EEG data.

`01-Analysis` contains scripts to analyze EEG data and a notebook to run statistics in R. 

`02-Figures` contains scripts and a notebook that steps through all the figures and statistics in the project.


## Reference

This project is described in the following preprint:


## Requirements

This project was written in Python 3 and requires Python >= 3.7 to run.

To re-run this project, you will need some external dependences.

Dependencies include 3rd party scientific Python packages:
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [scipy](https://github.com/scipy/scipy)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [statsmodels](https://github.com/statsmodels/statsmodels)
- [seaborn](https://github.com/mwaskom/seaborn)


You can get and manage these dependencies using the [Anaconda](https://www.anaconda.com/distribution/) distribution, which we recommend.

In addition, this project requires the following dependencies:

 - [mne](https://github.com/mne-tools/mne-python) >= 0.23.0
 - [fooof](https://github.com/fooof-tools/fooof) >= 1.0.0
 
You can install the extra required dependencies by running:

```
pip install mne, fooof
```

R scripts require:
- [lme4](https://github.com/lme4/lme4)
- [ciTools](https://github.com/jthaman/ciTools)



## Data

This project uses data protected under HIPAA and any identifying features have been removed. Data is not available at this time. 
