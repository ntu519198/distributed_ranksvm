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
    print '{0} num_machines'.format(sys.argv[0])

def plot_qw_fw(times, funs, pwaccs, output_prefix):
    color_list = ['k', 'r', 'g']
    legend_list = ['TreeTron', 'TreeTron-qw', 'TreeTron-fw']

    xlabel = 'Time (s)'
    ylabels = {'fun':'Relative function value difference', \
            'pwacc':'Relative pairwise accuracy difference (%)'}
    for ytype in ['fun', 'pwacc']:
        if ytype == 'fun':
            x = times
            y = funs
        else:
            x = times
            for idx, row in enumerate(x):
                x[idx] = x[idx][1:]
            y = pwaccs
        for i in xrange(0, len(x)):
            if ytype == 'fun':
                lines = plt.semilogy(x[i],y[i], color_list[i], label = legend_list[i])
            else:
                lines = plt.plot(x[i],y[i], color_list[i], label = legend_list[i])
            setp(lines, linewidth=4)
        max_x = 0.0
        for i, xx in enumerate(x):
            if min(max(xx),time_limit) > max_x:
                max_x = min(max(xx),time_limit)

        plt.xlim([0, max_x])
        if ytype == 'fun':
            plt.ylim([0, 1])
        else:
            min_y = 2147483647.0
            max_y = -min_y
            for yy in y:
                if yy[0] < min_y:
                    min_y = yy[0]
                if max(yy) > max_y:
                    max_y = max(yy)
            plt.ylim([min_y, max_y])
        plt.tick_params(labelsize=16)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabels[ytype], fontsize=20)
        plt.title('', fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig(figure_dir+output_prefix+'_'+str(nr_machines)+'_'+ytype+'.png',dpi=300,format='png')
        plt.clf()

def read_log(log_name):
    times = []
    funs = []
    pwaccs = []
    ndcgs = []
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

def main(nr_machines):
    print "Plotting TreeTron/TreeTron-QW/TreeTron_FW comparison:"
    for data in all_data:
        times = [] 
        funs = []
        pwaccs = []
        log_c = best_log_c[data]
        c = 2**log_c
        eps = 1e-6
        for log_dirname in [log_dir, log_QW_dir, log_FW_dir]:
            if log_dirname == log_dir:
                log_name = '{0}{1}_c{2}_e{3}.log'.format(log_dirname,data,c,eps)
            elif log_dirname == log_QW_dir:
                log_name = '{0}{1}_c{2}_e{3}_a.{4}.log'.\
                            format(log_dirname,data,c,eps,nr_machines)
            else:
                log_name = '{0}{1}_c{2}_e{3}.{4}.log'.\
                            format(log_dirname,data,c,eps,nr_machines)
            time, fun, pwacc = read_log(log_name)
            times.append(time)
            funs.append(fun)
            pwaccs.append(pwacc)
                
        output_prefix = data
        for i, fun in enumerate(funs):
            for j, f in enumerate(fun):
                funs[i][j] = (funs[i][j]-best_f[data])/best_f[data]
        for i, pwacc in enumerate(pwaccs):
            for j, pa in enumerate(pwacc):
                pwaccs[i][j] = (pwaccs[i][j]-best_pwacc[data])/best_pwacc[data]*100

        plot_qw_fw(times, funs, pwaccs, output_prefix)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit(print_usage())
    nr_machines = int(sys.argv[1])
    main(nr_machines)


