# Collections of scripts for preparing downscaling data

Yingkai Sha
2020-01-22

Data source:
* [PRISM](http://www.prism.oregonstate.edu/)
* [NCEP GDAS/FNL](https://rda.ucar.edu/datasets/ds083.3/)

Scripts are named as: `(data)_(functionality)_(variables)_(regions)_(time).py`

# PRISM scripts

Generating wget links (run `wget -i *.txt`)
* PRISM_wget-link-gen_NRT-vars.py

Converting raw files to HDF5: 
* PRISM_preprocess_clim-vars.py
* PRISM_preprocess_NRT-vars.py

File creations: 
* `PRISM_PCT_2015_2020.hdf`
* `PRISM_TMAX_2015_2020.hdf`
* `PRISM_TMIN_2015_2020.hdf`
* `PRISM_TMEAN_2015_2020.hdf`
* `PRISM_PCT_clim.hdf`
* `PRISM_TMAX_clim.hdf`
* `PRISM_TMIN_clim.hdf`
* `PRISM_TMEAN_clim.hdf`

Coarse-graining to 0.25-degree
* PRISM_regrid-0.25_clim-vars-etopo.py
* PRISM_regrid-0.25_NRT-vars-etopo.py

File creations:
* `PRISM_regrid_clim.hdf`
* `PRISM_regrud_2015_2020.hdf`

Feature engineering and generation
* PRISM_feature-gen_PCT_US.py
* PRISM_feature-gen_T2_US.py

File creations:
* `PRISM_PCT_features_2015_2020.hdf`
* `PRISM_TMAX_features_2015_2020.hdf`
* `PRISM_TMIN_features_2015_2020.hdf`
* `PRISM_TMEAN_features_2015_2020.hdf`

# FNL scripts
FNL scripts combines all preprocessing steps (raw file processing, regridding, feature generation).

* FNL_opt_all-vars-etopo_BC.py
* FNL_opt_all-vars-etopo_US.py

File creations
* `NCEP_FNL_BC_2016_2018_4km.hdf`
* `NCEP_FNL_2016_2018_4km.hdf`

# Pipeline scripts

Pipeline scripts perform random cropping, normalization and data augmentation for training DL models.

* Feature_pipeline_PCT_US.py
* Feature_pipeline_T2_US.py
