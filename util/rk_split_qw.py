#!/usr/bin/env python
import os, sys

def print_usage():
    print '{0} input_file_name output_file_prefix num_machine'.format(sys.argv[0])

def divide_nnz(res_nnz, num_machine):
    nnz_per_machine = int(res_nnz/num_machine)
    num_chunk = nnz_per_machine \
    if res_nnz*1.0/num_machine == nnz_per_machine \
    else nnz_per_machine+1

    return num_chunk

def rksvm_read_data(data_file_name):
    qid_list_dict = {}
    nnz_dict = {}
    total_nnz = 0
    total_n = 0
    for idx, line in enumerate(open(data_file_name)):
        label, query, features = line.split(None, 2)
        features_split = features.split()
        qid = int(query.split(':')[1])
        if qid not in qid_list_dict:
            qid_list_dict[qid] = [idx]
            nnz_dict[qid] = len(features_split)-1
        else:
            qid_list_dict[qid] += [idx]
            nnz_dict[qid] += len(features_split)-1
        total_nnz = total_nnz+len(features_split)-1
        for feature in features_split:
            if ':' not in feature:
                break
            ind, val = feature.split(':')
            if total_n < int(ind):
                total_n = int(ind)
    return (qid_list_dict, nnz_dict, total_nnz, total_n)

def rk_split_qw(src_path, dst_prefix, num_machine):
    src_dir, src_name = os.path.split(src_path)
    dst_path = dst_prefix

    [qid_list_dict, nnz_dict, total_nnz, total_n] = rksvm_read_data(src_path)
    nnz_per_machine = divide_nnz(total_nnz, num_machine)
    res_nnz = total_nnz
    res_num_machine = num_machine
    local_l_list = [] #Storing number of instances in each splitted file
    local_l = 0
    local_nnz = 0
    for qid in sorted(qid_list_dict.iterkeys()):
        local_l += len(qid_list_dict[qid])
        local_nnz += nnz_dict[qid]
        if local_nnz >= nnz_per_machine:
            local_l_list += [local_l]
            res_nnz -= local_nnz
            res_num_machine -= 1
            if res_num_machine == 0:
                break
            local_l = 0
            local_nnz = 0
            nnz_per_machine = divide_nnz(res_nnz, res_num_machine)

    nnz_per_machine = divide_nnz(total_nnz, num_machine)
    res_nnz = total_nnz
    res_num_machine = num_machine
    
    # Writing out splitted file with header lines (local_l and n)
    output_file = open(dst_prefix + '.{0}'.format(num_machine-res_num_machine), 'w')
    local_l = local_l_list[0]
    output_file.write('{0}\n{1}\n'.format(local_l, total_n))
    local_nnz = 0

    lines = open(src_path).readlines()
    for qid in sorted(qid_list_dict.iterkeys()):
        local_nnz += nnz_dict[qid]

        for idx in qid_list_dict[qid]:
            output_file.write(lines[idx])

        if local_nnz >= nnz_per_machine:
            output_file.close()
            res_nnz -= local_nnz
            res_num_machine -= 1
            if res_num_machine == 0:
                break
            output_file = open(dst_prefix + '.{0}'.\
                                format(num_machine-res_num_machine), 'w')
            local_l = local_l_list[num_machine-res_num_machine]
            output_file.write('{0}\n{1}\n'.format(local_l, total_n))
            local_nnz = 0
            nnz_per_machine = divide_nnz(res_nnz, res_num_machine)

def main():
    if len(sys.argv) != 4:
        exit(print_usage())
    src_path = sys.argv[1]
    dst_prefix = sys.argv[2]
    num_machine = int(sys.argv[3])
    rk_split_qw(src_path, dst_prefix, num_machine)

if __name__ == '__main__':
    main()
