# TDVQP_SM
This archive contains the code and data for the paper "Mixed Quantum Classical Dynamics for Near Term Quantum Computers".  This should allow the interested reader to reproduce all the figures and run their own simulations. 

The ipython notebooks are self-contained and have code to generate all the relevant figures in the paper. To reproduce the figures, all of the .zip files must be extracted so that the top-level directory is in the same folder as the ipython notebooks. Other simulations can be run by following the examples in the .slurm file, running the runme.py with appropriate inputs or by scripting around the pVQD_cluster.py file. The exac_simulator.py file contains the exact simulator that is used to compare the quantum computing results. 

The easiest way to generate results is to use the `runme.py` file with arguments. These are

"-n", which takes a string and is going to be the basename of the output files, you can also give standard linux paths as part of the name
"-r" is the number of grid points, 16 was used in the work and corresponds to 4 qubits
"-c" is the cost function, which can be either global (all 0 string) or local (generates #qubits cost functions which just flip one bit at a time). 
"-o" optimization algorithm, chosen to be 'sdg', but could be 'adam'. 
"-g" the computation of the gradient, leave to "param_shift"
"-rs", dest="restart" sets a restart flag in case you want to compute from a failed calculation or reuse the starting wavefunction, string of 'no' or filename of the restart file
"-t" size of timestep as a float representing atomic units
"-tr" order of Trotterization, 1 was used
"-s" number of steps in the time propagation desired, in the main text this was 1000. 
"-d" depth or layers of the chosen ansat
"-rf" setting the rf value of the SM model, 5 was used in this work
"-rl" rl value, 4 was used
"-rr" rr value, 3.2 was used
"-sh" number of shots, 0 is infinite shot limit
"-L"  distance between ions, set to 19 in the work
"-ths" threshold of the optimizer, set to 0.99999
"-x" initial position of the ion, set to -2.
"-v" initial velocity, set to 0.00114
"-i" initial state as either a list of integers, which begins the simulation at an equal superposition of those eigenstates, or as a normalized vector of floats, which begins with a weighted superposition of the eigenstates. 
"-p" flag for parameterized simulation, which decouples the classical and quantum systems by only using the ideal trajectory. 
"-pad" padding of the simulation, which is used to not have a 'wall' at the ions (particle in a box situation). This is set to 20. 
"-m" number of MD paths
"-b" qiskit backend. use statevector_simulator for infinite shot limit and qasm_simulator when using non-zero shots. 

To run this, the most critical requirement is to use qiskit 0.22.1. Other libraries are not expected to have much of an impact on performance, but the versions are provided below:

Specific library versions used in the generation of the original figures:
numpy 1.23.4
scipy 1.7.1
qiskit 0.22.1
seaborn 0.12.2
matplotlib 3.6.2
pandas 1.4.4
