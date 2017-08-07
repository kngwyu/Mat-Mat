#!/bin/bash
#PBS -q u-lecture
#PBS -Wgroup_list=gt19
#PBS -l select=8:mpiprocs=32
#PBS -l walltime=00:01:00

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh

mpirun ./mat-mat








