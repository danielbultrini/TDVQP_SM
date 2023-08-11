# TDVQP_SM
This archive contains the code and data for the paper "Mixed Quantum Classical Dynamics for Near Term Quantum Computers".  This should allow the interested reader to reproduce all the figures and run their own simulations. 

The ipython notebooks are self-contained and have code to generate all the relevant figures in the paper. To reproduce the figures, all of the .zip files must be extracted so that the top-level directory is in the same folder as the ipython notebooks. Other simulations can be run by following the examples in the .slurm file, running the runme.py with appropriate inputs or by scripting around the pVQD_cluster.py file. The exac_simulator.py file contains the exact simulator that is used to compare the quantum computing results. 

To run this, the most critical requirement is to use qiskit 0.22.1. Other libraries are not expected to have much of an impact on performance, but the versions are provided below:

Specific library versions used in the generation of the original figures:
numpy 1.23.4
scipy 1.7.1
qiskit 0.22.1
seaborn 0.12.2
matplotlib 3.6.2
pandas 1.4.4
