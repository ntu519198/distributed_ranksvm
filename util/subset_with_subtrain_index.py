#!/usr/bin/env python
import sys, subprocess, numpy as np

def print_usage():
    print '{0} src_name, sub_train_name, sub_test_name sub_train_index'.format(sys.argv[0])

def run_subset(src_name, sub_train_name, sub_test_name, sub_train_index):
    fp_r = open(src_name)
    fp_sub_tr = open(sub_train_name,'w')
    fp_sub_t = open(sub_test_name,'w')
    fp_sub_ti = open(sub_train_index)
    p = subprocess.Popen(['wc','-l',src_name],stdout=subprocess.PIPE)
    l = int(p.communicate()[0].split()[0])

    sub_train_set = set()
    for line in fp_sub_ti.readlines():
            if int(line.strip()) not in sub_train_set:
                    sub_train_set.add(int(line.strip()))
    idx = 0
    while True:
            line = fp_r.readline()
            if not line:
                    break
            if idx in sub_train_set:
                    fp_sub_tr.write(line)
            else:
                    fp_sub_t.write(line)
            idx += 1
    fp_r.close()
    fp_sub_tr.close()
    fp_sub_t.close()
    fp_sub_ti.close()
if __name__ == '__main__':
    if len(sys.argv) != 5:
        exit(print_usage())
    src_name = sys.argv[1]
    sub_train_name = sys.argv[2]
    sub_test_name = sys.argv[3]
    sub_train_index = sys.argv[4]
    run_subset(src_name, sub_train_name, sub_test_name, sub_train_index)

