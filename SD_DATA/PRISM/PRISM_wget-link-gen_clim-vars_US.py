'''
PRISM clim data
Generating download links + deleting old originals
source: http://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf
'''

# general tools
import sys
import subprocess
from glob import glob
from datetime import datetime, timedelta

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
from namelist import * 

# macros
vars = ['PCT', 'TMAX', 'TMIN', 'TMEAN']
del_old = True  # delete old files
del_txt = True  # delete old wget scripts

# wget script target path
dirs = {}
dirs['PCT']   = PRISM_CLIM_US_dir+'PCT/'
dirs['TMAX']  = PRISM_CLIM_US_dir+'TMAX/'
dirs['TMIN']  = PRISM_CLIM_US_dir+'TMIN/'
dirs['TMEAN'] = PRISM_CLIM_US_dir+'TMEAN/'

# PRISM server keywords
keywords = {} # 
keywords['PCT']   = 'ppt'
keywords['TMAX']  = 'tmax'
keywords['TMIN']  = 'tmin'
keywords['TMEAN'] = 'tmean'

for var in vars:
    print('===== Extracting {} ====='.format(var))
    
    if del_old:
        # delete old files
        cmd = 'rm -rf {}*[0-9]*'.format(dirs[var])
        print(cmd)
        subprocess.call(cmd, shell=True)
    if del_txt:
        # delete old wget script
        cmd = 'rm -f {}*wget*txt'.format(dirs[var])
        print(cmd)
        subprocess.call(cmd, shell=True)
    
    # base_link
    base_link = 'http://services.nacse.org/prism/data/public/normals/4km/{}/'.format(keywords[var])
    
    # wget link gen
    filename = 'wget_{}.txt'.format(var)
    print('Creating {}'.format(filename))
    f_io = open(dirs[var]+filename, 'w')
    for i in range(1, 13):
        # print(...)
        f_io.write(base_link+str(i)+'\n') # multi-lines
    f_io.close()









