#!/bin/bash

# Executable for parallel processing of PixelAV's *.gz files
# Takes as argument the integer in the file name
# For a new dataset, adjust EOSDIR

i=$1

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc8-opt/setup.sh

outdir=/save/dir/parquet_temp/
#point to the correct path
EOSDIR=/loc/to/files/dataset_XYZ/
# Booleans for offsetting clusters, adding noise, and applying a per-pixel charge threshold.
offcenter_data=true
add_noise=true
apply_charge_threshold=true
# NOTE: datagen.py contains global variables with sensor geometry and pixel array size (eg - 13x21) defined for a default 20 time slices. The global variables also have the noise and charge threshold parameters.

mkdir unflipped

xrdcp root://eosuser.cern.ch/$EOSDIR/pixel_clusters_d${i}.out.gz pixel_clusters_d${i}.out.gz
pwd
gunzip pixel_clusters_d${i}.out.gz

python datagen.py $i $offcenter_data $add_noise $apply_charge_threshold

xrdcp -f unflipped/labels_d${i}.parquet root://eosuser.cern.ch/$outdir/unflipped/labels_d${i}.parquet
xrdcp -f unflipped/recon2D_d${i}.parquet root://eosuser.cern.ch/$outdir/unflipped/recon2D_d${i}.parquet
xrdcp -f unflipped/recon3D_d${i}.parquet root://eosuser.cern.ch/$outdir/unflipped/recon3D_d${i}.parquet

if [ "$offcenter_data" = true ]; then
    xrdcp -f unflipped/recon2D_uncentered_d${i}.parquet root://eosuser.cern.ch/$outdir/unflipped/recon2D_uncentered_d${i}.parquet
    xrdcp -f unflipped/recon3D_uncentered_d${i}.parquet root://eosuser.cern.ch/$outdir/unflipped/recon3D_uncentered_d${i}.parquet
    xrdcp -f unflipped/offset_histogram_d${i}.png root://eosuser.cern.ch/$outdir/unflipped/offset_histogram_d${i}.png
fi
