#!/usr/bin/env python
from parameter import *
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from pylab import *

def print_usage():
    print '{0} QW/FW'.format(sys.argv[0])

def plot_speedup(x, y, data):
    lines = plt.plot(x, y, 'r')
    setp(lines, linewidth=4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))
    plt.xlabel('Number of machines', fontsize=20)
    plt.ylabel('Speedup', fontsize=20)
    plt.title('', fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(figure_speedup_dir+data+'_speedup.png',dpi=300,format='png')
    plt.clf()

# Get the smallest pairwise accuracy among TreeTron/TreeTron-QW/TreeTron-FW to 
# Be the target accuracy for measuring speedup
def get_final_pwacc(times, pwaccs):
    final_pwacc = 2147483647.0
    for i, pwacc in enumerate(pwaccs):
        if float(pwacc[-1]) < final_pwacc and times[i][-1] < time_limit:
            final_pwacc = float(pwacc[-1])
    return final_pwacc

# Get the time acheiving the target accuracy
def get_stop_times(times, pwaccs, final_pwacc):
    stop_times = []
    for i, time in enumerate(times):
        find_stop_time = False
        for j, sec in enumerate(time):
            if j == 0:
                continue
            if float(pwaccs[i][j-1]) >= final_pwacc:
                stop_times.append(time[j])
                find_stop_time = True
                break
        if find_stop_time == False:
            stop_times.append(time[len(time)-1])
    return stop_times

def read_log(log_name):
    times = []
    funs = []
    pwaccs = []
    is_first = True
    for idx, line in enumerate(open(log_name)):
        if 'iter' not in line:
            continue
        tokens = line.strip().split()
        times.append(float(tokens[-1]))
        funs.append(float(tokens[3]))
        if is_first:
            pwaccs.append(0)
            is_first = False
        else:
            pwaccs.append(float(tokens[5]))
    times = np.array(times)
    funs = np.array(funs)
    pwaccs = np.array(pwaccs[1:])
    return times, funs, pwaccs

def main():
    split_type = sys.argv[1]
    log_dir_list = ['log/', 'log_'+split_type+'/']
    nr_machine_list = [1,2,4,8,16]
    print "Plotting TreeTron-{0} speedup:".format(split_type)
    for data in all_data:
        if split_type != splits[data]:
            continue

        times = []
        funs = []
        pwaccs = []
        log_isread = False
        log_c = best_log_c[data]
        c = 2**log_c
        for eps in [1e-6]:
            for nr_machine in nr_machine_list:
                if nr_machine == 1:
                    log_dirname = log_dir
                    log_name = '{0}{1}_c{2}_e{3}.log'.format(log_dirname,data,c,eps)
                else:
                    log_dirname = log_dir_list[1]
                    if log_dirname == log_QW_dir:
                        log_name = '{0}{1}_c{2}_e{3}_a.{4}.log'.format\
                                        (log_dirname,data,c,eps,nr_machine)
                    else:
                        log_name = '{0}{1}_c{2}_e{3}.{4}.log'.format\
                                        (log_dirname,data,c,eps,nr_machine)

                time, fun, pwacc = read_log(log_name)
                times.append(time)
                pwaccs.append(pwacc)

            final_pwacc = get_final_pwacc(times, pwaccs)
            stop_times = get_stop_times(times, pwaccs, final_pwacc)

            xlim = (0, max(nr_machine_list))
            ylim = (0, stop_times[0]/min(stop_times))
            xlabel = 'Number of machines'
            ylabel = 'Speedup'
            title = ''
            output_file_name = '{0}_speedup.png'.format(data, c, eps)
            nr_machines = [np.array(xrange(0,len(nr_machine_list)))]
            single_machine_time = stop_times[0]

            for i, stop_time in enumerate(stop_times):
                stop_times[i] = single_machine_time/stop_times[i]
            stop_times = [np.array(stop_times)]
            x = nr_machines[0]
            y = stop_times[0]
            plot_speedup(x, y, data)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit(print_usage())
    main()


