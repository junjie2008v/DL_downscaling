import os
import h5py
import pygrib
import subprocess
import numpy as np
from glob import glob
from os.path import isfile, basename
from datetime import datetime, timedelta

# work flow: copy from source --> unzip --> process and save
team_drive = '/glade/scratch/ksha/DATA/NAEFS/'      # source dir
my_drive = '/glade/scratch/ksha/DATA/NAEFS_TEMP/'   # temporal space for un-ziping files
local_space = '/glade/scratch/ksha/DATA/FILE_TEMP/' # storage of post-processed data

date_list = [datetime(2017, 1, 1, 0, 0)+timedelta(days=i) for i in range(365)] # time period
skip_id = np.array([2, 3, 11, 12, 21, 22, 23, 24, 41, 42]) # ignoring variables by grib table index 

boss1 = 'cmc' # NAEFS Candian member (cmc v.s. ncep)
N_member = 21 # number of Canadian members (1 base run + 20 perturbation runs)
# selecting fcst horizons
tail_str = ['f006', 'f012', 'f018', 'f024', 'f030', 'f036', 'f042', 'f048', 'f054', 'f060', 'f066', 'f072']
L_check = len(tail_str)

for date in date_list:
    filename = datetime.strftime(date, '%Y%m%d%H.tgz') # filename: 2017010100.tgz
    print('Working on file: '+filename)
    temp_dir1 = team_drive+filename
    temp_dir2 = local_space+filename
    temp_dir3 = local_space+'temp2/'+filename[:10]+'/' # temp dir for tar cmd
    if isfile(temp_dir1):
        # gen shell cmds
        cmd_cp = 'cp "'+temp_dir1+'" "'+temp_dir2+'"'
        cmd_tar = 'tar xvzf "'+temp_dir2+'" -C "'+local_space+'temp2/"'
        cmd_mv = 'mv '+local_space+'*hdf "'+my_drive+'"'
        cmd_rm1 = 'rm -rf "'+temp_dir2+'"'
        cmd_rm2 = 'rm -rf '+local_space+'temp2/*'
        # run shell cmds
        subprocess.run(cmd_cp, shell=True)
        print('\tUnpacking tar file')
        subprocess.run(cmd_tar, shell=True)
        print('\tExtracting variables')
        # Collecting features
        FEATUREs = np.zeros((N_member, L_check, 53, 90, 180))*np.nan # 53 because i know
        for i in range(N_member):
            print('Member {}'.format(i))
            grb_names = sorted(glob(temp_dir3+'*'+boss1+'*{0:02d}*.t00z.pgrb2f0[0,1,2,3,4,5,6,7]*'.format(i))) # names of Canadian member single files           
            for j, grb_name in enumerate(grb_names):
                key = basename(grb_name)[-4:]
                ind_list = [i for i, s in enumerate(tail_str) if key in s] # selecting by fcst horizon
                # checking if empty file
                if len(ind_list)>0 and os.stat(grb_name).st_size > 1000000:
                    print(grb_name)
                    ind = ind_list[0]
                    # grib2 cmds
                    grbs = pygrib.open(grb_name)
                    count = 0
                    for k, var_name in enumerate(grbs.select()):
                        if np.logical_not(k in skip_id):
                            FEATUREs[i, ind, count, ...] = grbs.select()[k].values[91:, 180:]
                            count += 1
                    grbs.close()
        # save as hdf5
        out_name = 'NAEFS_'+filename[:8]+'_CMC_72H.hdf'
        print('\tSaving as HDF: '+out_name)
        hdf_obj = h5py.File(local_space+out_name, 'w')
        hdf_obj.create_dataset('FEATURE_CMC', data=FEATUREs)
        #hdf_obj.create_dataset('var_names', (53, 1), 'S10', data=str_encode(cmc_varname))
        hdf_obj.close()
        # clean unzipped files to free space
        print('\tBackup output to storage')
        subprocess.run(cmd_mv, shell=True)
        print('\tClean-up temp folder')
        subprocess.run(cmd_rm1, shell=True)
        subprocess.run(cmd_rm2, shell=True)
    else:
        print('\tFile '+filename+' does not exist.')
