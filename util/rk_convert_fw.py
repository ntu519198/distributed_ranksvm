#!/usr/bin/env python
import sys
import os
import collections
import scipy
from scipy.sparse import *
from scipy import *

def print_usage():
    print '{0} svm_file_name'.format(sys.argv[0])

def rksvm_read_csr(data_file_name):
    prob_y = []
    prob_qid = []
    val_list = []
    col_ind = []
    row_ptr = []

    l = 0
    n = 0
    for idx, line in enumerate(open(data_file_name)):
        line = line.split(None, 2)

        # In case an instance with all zero features
        if len(line) == 2: line += ['']
        label, query, features = line

        cmt_idx = features.find('#')
        if cmt_idx != -1:
            features = features[:cmt_idx]+'\n'
        qid = query.split(":")[1]

        for e in features.split():
            ind, val = e.split(":")
            ind = int(ind)
            col_ind.append(ind - 1)
            val_list.append(float(val))
            row_ptr.append(idx)
            if ind > n: 
                n = ind
        prob_y += [float(label)]
        prob_qid += [int(qid)]
    l = idx + 1
    return (prob_y, prob_qid, val_list, col_ind, row_ptr, l, n)

def rk_convert_fw(src_name, dst_name):
    y, qid, val, col_ind, row_ptr, l, n = rksvm_read_csr(src_name)

    y = scipy.array(y)
    qid = scipy.array(qid)
    val = scipy.array(val)
    col_ind = scipy.array(col_ind)
    row_ptr = scipy.array(row_ptr)

    csr = csr_matrix((val, (row_ptr, col_ind)), shape = (l, n))

    csc = csr.tocsc()
    csc_indices = csc.indices
    csc_indptr = csc.indptr
    csc_data = csc.data
    with open(dst_name, 'w') as output_file:
        for item in y:
            output_file.write('{0} '.format(item))
        output_file.write('\n')

        for item in qid:
            output_file.write('{0} '.format(item))
        output_file.write('\n')

        idx = 1
        pre_csc_indptr = csc_indptr[0]
        for ind, item in enumerate(csc_data):
            while ind >= csc_indptr[idx]:
                if csc_indptr[idx] == pre_csc_indptr:
                    output_file.write('0 \n')
                else:
                    pre_csc_indptr = csc_indptr[idx]
                    output_file.write('\n')
                idx += 1

            if ind == pre_csc_indptr:
                output_file.write('{0}:{1} '.format(csc_indices[ind] + 1, item))
            else:
                output_file.write('{0}:{1} '.format(csc_indices[ind] + 1, item))
        output_file.write('\n')

def main():
    src_name = sys.argv[1]
    dst_name = sys.argv[1] + '.fw'
    rk_convert_fw(src_name, dst_name)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit(print_usage())
    main()
