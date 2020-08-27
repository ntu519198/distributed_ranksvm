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

def rksvm_read_data_fw(data_file_name):
    nnz_list = [] 
    label_line = ''
    qid_line = ''
    total_nnz = 0
    total_l = 0
    for idx, line in enumerate(open(data_file_name)):
        if idx == 0:
            label_line = line
            continue
        if idx == 1:
            qid_line = line
            continue

        instances = line
        instances_split = instances.split()
        for instance in instances_split:
            if len(instance) == 1:
                continue
            ind, val = instance.split(':')
            if int(ind) > total_l:
                total_l = int(ind)

        nnz_list.append(len(instances_split))
        total_nnz = total_nnz+len(instances_split)
    return (label_line, qid_line, nnz_list, total_nnz, total_l)

def rk_split_fw(src_path, dst_prefix, num_machine):
    src_dir, src_name = os.path.split(src_path)
    dst_path = dst_prefix

    [label_line, qid_line, nnz_list, total_nnz, total_l] = rksvm_read_data_fw(src_path)
    nnz_per_machine = divide_nnz(total_nnz, num_machine)
    res_nnz = total_nnz
    res_num_machine = num_machine

    feat_per_machine = []
    local_nnz = 0
    for idx, nnz in enumerate(nnz_list):
        local_nnz += nnz
        if local_nnz >= nnz_per_machine:
            res_nnz -= local_nnz
            res_num_machine -= 1
            if res_num_machine == 0:
                feat_per_machine.append(len(nnz_list)-1)
                break
            nnz_per_machine = divide_nnz(res_nnz, res_num_machine)
            feat_per_machine.append(idx)
            local_nnz = 0
            continue

    lines = open(src_path).readlines()[2:] #Skip header lines (label and qid)
    for idx, feat_idx in enumerate(feat_per_machine):
        output_file = open(dst_path+'.{0}'.format(idx), 'w')

        if idx == 0:
            local_n = feat_idx+1
        else:
            local_n = feat_idx-feat_per_machine[idx-1]

        output_file.write(str(total_l)+'\n'+str(local_n)+'\n')
        output_file.write(label_line)
        output_file.write(qid_line)

        if idx == 0:
            for i in xrange(0, feat_idx+1):
                output_file.write(lines[i])
        else:
            for i in xrange(feat_per_machine[idx-1]+1, feat_idx+1):
                output_file.write(lines[i])

        output_file.close()

def main():
    src_path = sys.argv[1]
    dst_prefix = sys.argv[2]
    num_machine = int(sys.argv[3])
    rk_split_fw(src_path, dst_prefix, num_machine)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        exit(print_usage())
    main()

