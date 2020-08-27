#!/usr/bin/env python
"""
This script is modified from split.py in
the script of mpi liblinear by Yong Zhuang, et al.
"""
from parameter import *
import sys, subprocess, uuid, os, math, shutil

if len(sys.argv) != 4 and len(sys.argv) != 5:
    print('usage: {0} QW/FW machinefile svm_file [split_svm_file]'.format(sys.argv[0]))
    sys.exit(1)
split_type, machinefile, src_path = sys.argv[1:4]

if split_type != "QW" and split_type != "FW":
    print('usage: {0} QW/FW machines svm_file [split_svm_file]'.format(sys.argv[0]))
    sys.exit(1)

machines = set()
for line in open(machinefile):
    machine = line.strip()
    if machine in machines:
        print('Error: duplicated machine {0}'.format(machine))
        sys.exit(1)
    machines.add(machine)

machines = [] # To make sure the order of machine for feature-wise split
for line in open(machinefile):
    machine = line.strip()
    machines.append(machine)

nr_machines = len(machines)

src_basename = os.path.basename(src_path)
src_dirname = os.path.dirname(src_path)
if len(sys.argv) == 5:
    dst_path = sys.argv[4]
else:
    #dst_path = '{0}.sub'.format(src_basename)
    if split_type == "QW":
        dst_path = '{0}.{1}.sub'.format(src_path, nr_machines)
    else:
        dst_path = '{0}.fw.{1}.sub'.format(src_path, nr_machines)

#cmd = 'wc -l {0}'.format(src_path)
#p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
#nr_instances = int(p.stdout.read().strip().split()[0])
#p.communicate()

while True:
    temp_dir = '{1}tmp_{0}'.format(uuid.uuid4(), data_dir)
    if not os.path.exists(temp_dir): break
os.mkdir(temp_dir)

print('Spliting data...')
nr_digits = int(math.log10(nr_machines))+1
#cmd = 'split -l {0} --numeric-suffixes -a {1} {2} {3}.'.format(
#          int(math.ceil(float(nr_instances)/nr_machines)), nr_digits, src_path,
#          os.path.join(temp_dir, src_basename))
if split_type == "QW":
    cmd = 'util/rk_split_qw.py {0} {1} {2}'.\
            format(src_path, os.path.join(temp_dir, src_basename), nr_machines)
else:
    cmd = ""
    #if not os.path.exists(src_path+".fw"):
    cmd += 'util/rk_convert_fw.py {0} && '.format(src_path)
    cmd += 'util/rk_split_fw.py {0}.fw {1}.fw {2}'.\
            format(src_path, os.path.join(temp_dir, src_basename), nr_machines)
p = subprocess.Popen(cmd, shell=True)
p.communicate()

for i, machine in enumerate(machines):
    if split_type == "QW":
        temp_path = os.path.join(temp_dir, src_basename + '.' +
                                 str(i).zfill(nr_digits))
    else:
        temp_path = os.path.join(temp_dir, src_basename + '.fw.' +
                                 str(i).zfill(nr_digits))

    if machine == '127.0.0.1' or machine == 'localhost':
        cmd = 'mv {0} {1}'.format(temp_path, dst_path)
    else:
        cmd = 'ssh {0} "mkdir -p {1}";'.format(machine, os.path.join(os.getcwd(),src_dirname))
        cmd += 'scp {0} {1}:{2}'.format(temp_path, machine,
                                       os.path.join(os.getcwd(), dst_path))
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.communicate()
    print('The subset of data has been copied to {0}'.format(machine))

shutil.rmtree(temp_dir)
