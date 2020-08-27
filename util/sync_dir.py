#!/usr/bin/env python
import os, sys, subprocess, threading
from subprocess import PIPE
def print_usage():
    print '{0} sync_dir dst_path machinefile'.format(sys.argv[0])

def run_cmd(cmd):
	print cmd
	p = subprocess.Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
	p.communicate()

def sync_dir(sync_dir_name, dst_path, machinefile):
	sync_dir_abspath = os.path.abspath(sync_dir_name)
	thread_list = []
	for idx, line in enumerate(open(machinefile)):
		#skip master
		if idx == 0:
			continue
		ip = line.strip()
		sync_cmd = "scp -r {0} {1}:{2}".format(sync_dir_abspath, ip, dst_path)

		t = threading.Thread(target=run_cmd, args=[sync_cmd])
		thread_list.append(t)

	for t in thread_list:
		t.start()
	for t in thread_list:
		t.join()
if __name__ == '__main__':
	if len(sys.argv) != 4:
		exit(print_usage())

	sync_dir_name = sys.argv[1]
	dst_path = sys.argv[2]
	machinefile = sys.argv[3]
	sync_dir(sync_dir_name, dst_path, machinefile)
