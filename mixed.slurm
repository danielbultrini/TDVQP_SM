#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks-per-node=40
#SBATCH --time=5:30:30
#SBATCH --mem=40gb
#SBATCH --signal=B:TERM@180
#SBATCH --output=R-o-cpu%x.%j.out
#SBATCH --error=R-e-cpu%x.%j.err

trap 'echo signal recieved in BATCH!; kill -15 "${PID}"; wait "${PID}";' SIGINT SIGTERM
source ~/.bashrc
conda activate 310
for value in {0..99}; 
              do python /home/hd/hd_hd/hd_hq183/code/TDVQE/runme.py -n /home/hd/hd_hd/hd_hq183/code/TDVQE/results/mixed/mixed_$value -x -2 -v 0.2 -rf 5 -rl 4 -rr 3.1 -L 19 -p 0 -d 4 -tr 2 -pad 5 -i 0 1 -s 700 & 
              done
              wait
PID="$!"
wait "${PID}"
