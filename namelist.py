# ===== PATH ===== #
BACKUP_dir = '/glade/scratch/ksha/BACKUP/' # hdf file path

# Processed PRISM file path
PRISM_dir = BACKUP_dir + 'PRISM/'

# Raw PRISM file path
PRISM_PCT_dir   = PRISM_dir + 'OSU_PRISM_NRT_PCT/'
PRISM_TMAX_dir  = PRISM_dir + 'OSU_PRISM_NRT_TMAX/'
PRISM_TMIN_dir  = PRISM_dir + 'OSU_PRISM_NRT_TMIN/'
PRISM_TMEAN_dir = PRISM_dir + 'OSU_PRISM_NRT_TMEAN/'
PRISM_CLIM_US_dir = PRISM_dir + 'OSU_PRISM_normals/'
PRISM_CLIM_BC_dir = PRISM_dir + 'PCIC_PRISM_normals/' # <-- 800-m grids

# NCEP GDAS/FNL
NCEP_dir = BACKUP_dir + 'NCEP_FNL/'
NCEP_PCT_dir = NCEP_dir + 'TMAX-TMIN-PCT/'
NCEP_TMAX_dir = NCEP_dir + 'TMAX-TMIN-PCT/'
NCEP_TMIN_dir = NCEP_dir + 'TMAX-TMIN-PCT/'
NCEP_TMEAN_dir = NCEP_dir + 'TMEAN/'

# JRA 55
JRA_dir = BACKUP_dir + 'JRA55/'
JRA_TMEAN_dir = JRA_dir + 'TMEAN/'
JRA_TAIR_dir = JRA_dir + 'TAIR/'
JRA_SFP_dir = JRA_dir + 'SFP/'

# ERA
ERA_dir = BACKUP_dir + 'ERA_Interim/'
ERA_TMEAN_dir = ERA_dir + 'TMEAN/'
ERA_TAIR_dir = ERA_dir + 'TAIR/'
ERA_SFP_dir = ERA_dir + 'SFP/'

# MPAS
MPAS_dir = BACKUP_dir + 'MPAS/'

# NAEFS
NAEFS_dir = BACKUP_dir + 'NAEFS/'

# CFSR
CFSR_dir = BACKUP_dir + 'CFSR/TMEAN_1deg/'

# GFS
GFS_dir = BACKUP_dir + 'GFS/'

# obs
OBS_dir = BACKUP_dir + 'GHCN/'
OBS_CSV_dir = OBS_dir + 'csv/'

# Traning set path
BATCH_dir = '/glade/scratch/ksha/DATA/PRISM_dscale/'
# output figure path
fig_dir = '/glade/u/home/ksha/figures/'
# model, evaluation data path
save_dir = '/glade/work/ksha/data/Keras/PRISM_publish/'
# temporal model path
temp_dir = '/glade/work/ksha/data/Keras/BACKUP/'

# ===== Param ===== #
# subsetting from US to US west
subset_ind = [0, 600, 0, 600]

# interp. method
interp_method = 'cubic'

# CNN training batch size
batch_size = 200

# tuning/transferring domain inds
ind_tune = 408; ind_trans = 504

# bicubic interpolation on PCT brings negative values, use thres to clean up
#    *after log transformation
thres = 2.5e-1

