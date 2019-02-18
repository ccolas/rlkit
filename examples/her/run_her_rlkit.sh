#!/bin/sh
#SBATCH --mincpus 5
#SBATCH -p longq
#SBATCH -t 24:00:00
#SBATCH -e run_her_rlkit.sh.err
#SBATCH -o run_her_rlkit.sh.out
rm log.txt; 
export EXP_INTERP='/cm/shared/apps/intel/composer_xe/python3.5/intelpython3/bin/python3' ;
$EXP_INTERP her_tsac_gym_fetch_reach.py &
$EXP_INTERP her_tsac_gym_fetch_reach.py &
$EXP_INTERP her_td3_gym_fetch_reach.py &
$EXP_INTERP her_td3_gym_fetch_reach.py &
wait