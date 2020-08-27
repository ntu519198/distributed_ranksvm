#!/usr/bin/env python
import csv
import os, sys

nr_feat = -1
feat_dict_list = []

def print_usage():
    print "{0} train_csv_path test_csv_path".format(sys.argv[0])

def build_feat_map(train_csv_path, test_csv_path):
    global nr_feat
    global feat_dict_list
    #with open(path+train_csv_path['AMAZON']) as csvfile:
    with open(train_csv_path) as csvfile:
        train_reader = csv.reader(csvfile, delimiter=',')
        header = next(train_reader)
        nr_feat = len(header)-1
        feat_dict_list = [{} for i in xrange(0, nr_feat)]
        for row in train_reader:
            for idx, attr in enumerate(row[1:]):
                feat_dict_list[idx][int(attr)] = 1
    csvfile.close()
    #with open(path+test_csv_path['AMAZON']) as csvfile:
    with open(test_csv_path) as csvfile:
        test_reader = csv.reader(csvfile, delimiter=',')
        header = next(test_reader)
        for row in test_reader:
            for idx, attr in enumerate(row[1:]):
                feat_dict_list[idx][int(attr)] = 1
    csvfile.close()
    j = 1
    for i in xrange(0, nr_feat):
        for key in sorted(feat_dict_list[i].iterkeys()):
            feat_dict_list[i][key] = j
            j += 1

def gen_feat(train_csv_path, test_csv_path):
    global nr_feat
    global feat_dict_list
    output_dir = os.path.dirname(test_csv_path)
    #fp_train_out = open(path+'AMAZON/'+'train', 'w')
    #fp_test_out = open(path+'AMAZON/'+'test', 'w')
    fp_train_out = open(os.path.join(output_dir,'train'), 'w')
    fp_test_out = open(os.path.join(output_dir,'test'), 'w')
    #with open(path+train_csv_path['AMAZON']) as csvfile:
    with open(train_csv_path) as csvfile:
        train_reader = csv.reader(csvfile, delimiter=',')
        header = next(train_reader)
        for row in train_reader:
            fp_train_out.write(str(row[0]))
            for idx, attr in enumerate(row[1:]):
                fp_train_out.write(" "+str(feat_dict_list[idx][int(attr)])+":1")
            fp_train_out.write("\n")
    csvfile.close()
    #with open(path+test_csv_path['AMAZON']) as csvfile:
    with open(test_csv_path) as csvfile:
        test_reader = csv.reader(csvfile, delimiter=',')
        header = next(test_reader)
        for row in test_reader:
            fp_test_out.write("1 ")
            for idx, attr in enumerate(row[1:]):
                fp_test_out.write(" "+str(feat_dict_list[idx][int(attr)])+":1")
            fp_test_out.write("\n")
    csvfile.close()
    fp_train_out.close()
    fp_test_out.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit(print_usage())
    train_csv_path = sys.argv[1]
    test_csv_path = sys.argv[2]
    build_feat_map(train_csv_path, test_csv_path)
    gen_feat(train_csv_path, test_csv_path)
