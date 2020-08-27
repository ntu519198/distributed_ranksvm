#!/usr/bin/env python
from parameter import *
import os, sys
def print_usage():
    print '{0} machinefile'.format(sys.argv[0])
def prepare_dist_data(nr_machines, machinefile):
    if nr_machines == 1:
		return

    print "Preparing distributed data ("+str(nr_machines)+" machines):"
    for data in all_data:
        for split in ["QW", "FW"]:
            cmd = "./rk_split.py {0} {1} {2}{3}".format(split, machinefile, data_dir, train_path[data])
            print cmd
            os.system(cmd)

def run_exp(nr_machines, machinefile):
    print "Running experiments ("+str(nr_machines)+" machines):"
    eps = 1e-6
    for data in all_data:
        if splits[data] == "QW":
            split_type = 0
        else:
            split_type = 1

        # Run TreeTron-QW and TreeTron-FW
        if nr_machines > 1:
            for split_type in [0, 1]:
                if split_type == 0:
                    log_name = "{0}_c{2}_e{3}_a.{1}.log".\
                                format(data, nr_machines, 2**best_log_c[data],eps)
                    cmd = "mpirun -n 2 --machinefile {1} {2} -S {3} -c {4} -e {5} -a -t {7}{6} {7}{8}.{0}.sub /dev/null | tee {9}{10}".\
                        format(nr_machines, machinefile, train_exe, split_type, 2**best_log_c[data], eps, test_path[data], data_dir, train_path[data], log_QW_dir, log_name)
                else:
                    log_name = "{0}_c{2}_e{3}.{1}.log".\
                                format(data, nr_machines, 2**best_log_c[data], eps)
                    cmd = "mpirun -n 2 --machinefile {1} {2} -S {3} -c {4} -e {5} -t {7}{6} {7}{8}.fw.{0}.sub /dev/null | tee {9}{10}".\
                        format(nr_machines, machinefile, train_exe, split_type, 2**best_log_c[data], eps, test_path[data], data_dir, train_path[data], log_FW_dir, log_name)
                print cmd
                os.system(cmd)
        else:
            # Run TreeTron
            log_name = "{0}_c{1}_e{2}.log".format(data, 2**best_log_c[data],eps)
            cmd = "{0} -c {1} -e {2} -t {4}{3} {4}{5} /dev/null | tee {6}{7}".\
                format(train_exe, 2**best_log_c[data], eps, test_path[data], data_dir, train_path[data], log_dir, log_name)
            print cmd
            os.system(cmd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit(print_usage())

    machinefile = sys.argv[1]
    nr_machines = len(open(machinefile).readlines())
    prepare_dist_data(nr_machines, machinefile)
    run_exp(nr_machines, machinefile)
