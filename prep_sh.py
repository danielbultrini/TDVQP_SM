
start = """#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks-per-node=40
#SBATCH --time=5:30:30
#SBATCH --mem=40gb
#SBATCH --signal=B:TERM@180
#SBATCH --output=R_%x.%j.out
#SBATCH --error=R_%x.%j.err

trap 'echo signal recieved in BATCH!; kill -15 "${PID}"; wait "${PID}";' SIGINT SIGTERM
source ~/.bashrc
conda activate 310
"""
# # FOR VARIATIONS OF ONE RUN
# with open('long_vqd.slurm',"w") as fil:
#     fil.write(start)
#     fil.write("""for value in {0..99}; 
#               do python /home/hd/hd_hd/hd_hq183/code/TDVQE/runme.py -n /home/hd/hd_hd/hd_hq183/code/TDVQE/results/short$value -x -2 -v 0.2 -rf 5 -rl 4 -rr 3.1 -L 19 -p 0 -d 4 -tr 2 -pad 5 -i 0 -s 10 & 
#               done
#               wait\n""")
#     fil.write("""PID="$!"\nwait "${PID}"\n""")

# # FOR particular values
with open('trotter_depth.slurm','w') as fil:
    fil.write(start)
    for rep in range(10):
        for depth in [3,4,5]:
            for trot in [1,2,3,4,10]:
                fil.write(f"""python /home/hd/hd_hd/hd_hq183/code/TDVQE/runme.py -n /home/hd/hd_hd/hd_hq183/code/TDVQE/results/depth_trotter_an/d_{depth}_tr_{trot}_rep{rep} -x -2 -v 0.2 -rf 5 -rl 4 -rr 3.1 -L 19 -p 0 -d {depth} -tr {trot} -pad 5 -i 0 -s 1000 &\n""")
    fil.write('wait\n')
    fil.write("""PID="$!"\nwait "${PID}"\n""")
        