#!/usr/bin/env python
import os,sys, numpy as np
def print_usage():
    print '{0} data_file_name num_qid'.format(sys.argv[0])

def divide_nnz(res_nnz, num_qid):
    nnz_per_qid = int(res_nnz/num_qid)
    num_chunk = nnz_per_qid \
    if res_nnz*1.0/num_qid == nnz_per_qid \
    else nnz_per_qid+1

    return num_chunk

def rksvm_read_data(data_file_name):
    total_nnz = 0
    for idx, line in enumerate(open(data_file_name)):
        line = line.split(None, 1)
        label, features = line
        features_list = features.split()
        total_nnz += len(features_list)
    return total_nnz

def add_qid(data_file_name,num_qid):
    total_nnz = rksvm_read_data(data_file_name)
    fp_r = open(data_file_name)
    fp_o = open(data_file_name+".{0}qids".format(num_qid),"w")
    res_nnz = total_nnz
    res_num_qid = num_qid
    nnz_per_qid = divide_nnz(res_nnz, res_num_qid)
    local_nnz = 0
    qid_idx = 1
    while True:
        line = fp_r.readline()
        if not line:
            break
        label, features = line.strip().split(None,1)
        feature_list = features.split()
        local_nnz += len(feature_list)
        if local_nnz >= nnz_per_qid:
            fp_o.write(label+' qid:{0} '.format(str(qid_idx))+features+'\n')
            res_nnz -= local_nnz
            res_num_qid -= 1
            qid_idx += 1
            local_nnz = 0
            if res_num_qid > 0:
                nnz_per_qid = divide_nnz(res_nnz, res_num_qid)
        else:
            fp_o.write(label+' qid:{0} '.format(str(qid_idx))+features+'\n')
    fp_r.close()
    fp_o.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        exit(print_usage())
    data_file_name = sys.argv[1]
    num_qid = int(sys.argv[2])
    add_qid(data_file_name, num_qid)
