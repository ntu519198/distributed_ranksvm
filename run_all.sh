# Copy & Compile on all machines (default 2 machines)
mpirun -n 2 --machinefile conf/machinefile02 mkdir -p `pwd`
util/sync_dir.py "`pwd`/solver" `pwd` conf/machinefile02
mpirun -n 2 --machinefile conf/machinefile02 make -C solver
# Generate data set
./gen_data.py
# Run experiments (default 2 machines)
./run_exp.py conf/machinefile02
# Run experiments for single machine (for comparison)
./run_exp.py conf/machinefile01

# Plot TreeTron/TreeTron-QW/TreTron-FW comparison
./plot_qw_fw_comparison.py 2

# Run experiments for 4, 8, 16 machines
#./run_exp.py conf/machinefile04
#./run_exp.py conf/machinefile08
#./run_exp.py conf/machinefile16

# Plot speedup 
#./plot_speedup QW
#./plot_speedup FW

