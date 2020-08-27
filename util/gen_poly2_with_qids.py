#!/usr/bin/env python
from math import sqrt
import sys

def print_usage():
    print "{0} svm_file gamma".format(sys.argv[0])


def gen_poly2_with_qids(src_name, gamma_str):

    fp_r = open(src_name)
    fp_o = open(src_name+'.poly2_g'+gamma_str,'w')
    lines = fp_r.readlines()
    ind_dict = {}
    max_ind = 0
    gamma = float(gamma_str)
    sqrt2 = sqrt(2.0)
    sqrt2_g = sqrt(2.0*gamma)
    for line in lines:
            label, query, features = line.strip().split(None,2)
            feature_list = features.split()
            ind_list = []
            for feature in feature_list:
                    if ':' not in feature:
                            continue
                    ind, val = feature.split(':')
                    ind = int(ind)
                    if ind not in ind_dict:
                            ind_dict[ind] = 1
                    ind_list.append(ind)
                    if ind > max_ind:
                            max_ind = ind
            for ind in ind_list:
                    for ind2 in ind_list:
                            if ind > ind2:
                                    continue
                            tup = (ind, ind2)
                            if tup not in ind_dict:
                                    ind_dict[tup] = 1

    for idx, ind_key in enumerate(sorted(ind_dict.iterkeys())):
            ind_dict[ind_key] = idx+1
    ind_dict['bias'] = idx+2

    for line in lines:
            label, query, features = line.strip().split(None,2)
            feature_list = features.split()
            ind_list = []
            val_list = []
            feat_dict = {}
            for feature in feature_list:
                    if ':' not in feature:
                            continue
                    ind, val = feature.split(':')
                    ind = int(ind)
                    val = float(val)
                    ind_list.append(ind)
                    val_list.append(val)
                    feat_dict[ind_dict[ind]] = sqrt2_g*val
            for i, ind in enumerate(ind_list):
                    for j, ind2 in enumerate(ind_list):
                            tup = (ind, ind2)
                            if i > j:
                                    continue
                            if i == j:
                                    feat_dict[ind_dict[tup]] = gamma*val_list[i]*val_list[j]
                            else:
                                    feat_dict[ind_dict[tup]] = sqrt2*gamma*val_list[i]*val_list[j]
            feat_dict[ind_dict['bias']] = 1
            feat_list = []
            for key in sorted(feat_dict.iterkeys()):
                    feat_list.append(str(key)+':'+str(feat_dict[key]))

            fp_o.write(label+" "+query+" "+" ".join(feat_list)+'\n')

    fp_r.close()
    fp_o.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit(print_usage())
    src_name = sys.argv[1]
    gamma_str = sys.argv[2]
    gen_poly2_with_qids(src_name, gamma_str)
