#!/usr/bin/env python
data_dir = "data/"
conf_dir = "conf/"
log_QW_dir = "log_QW/"
log_FW_dir = "log_FW/"
log_dir = "log/"
figure_dir = "figure/"
figure_speedup_dir = "figure_speedup/"
all_data = ['MSLR','YAHOO_SET1','MQ2007-list','MQ2008-list', 'MQ2007', 'MQ2008', 'AMAZON_poly2']
train_path = {"MQ2007":"MQ2007/Fold1/train.txt","MQ2008":"MQ2008/Fold1/train.txt","MSLR":"Fold1/train.txt.scale","YAHOO_SET1":"YAHOO/set1.train.txt.scale","YAHOO_SET2":"YAHOO/set2.train.txt.scale","MQ2007-list":"MQ2007-list/Fold1/train.txt","MQ2008-list":"MQ2008-list/Fold1/train.txt","AMAZON_poly2":"AMAZON/subtrain.256qids.poly2_g1"}
test_path = {"MQ2007":"MQ2007/Fold1/test.txt","MQ2008":"MQ2008/Fold1/test.txt","MSLR":"Fold1/test.txt.scale","YAHOO_SET1":"YAHOO/set1.test.txt.scale","YAHOO_SET2":"YAHOO/set2.test.txt.scale","MQ2007-list":"MQ2007-list/Fold1/test.txt","MQ2008-list":"MQ2008-list/Fold1/test.txt","AMAZON_poly2":"AMAZON/subtest.256qids.poly2_g1"}

splits = {"MQ2007":"QW","MQ2008":"QW","MSLR":"QW","YAHOO_SET1":"QW","YAHOO_SET2":"QW","MQ2007-list":"QW","MQ2008-list":"QW","AMAZON_poly2":"FW"}
best_log_c = {"MQ2007":-5,"MQ2008":8,"MSLR":8,"YAHOO_SET1":3,"YAHOO_SET2":-11,"MQ2007-list":-12,"MQ2008-list":-14,"AMAZON_poly2":-11}
best_log_g = {"AMAZON": 0}
best_f = {"MQ2007":5944.34998521, "MQ2008":7565689.54632,"MSLR_F1":0.615967, "YAHOO_SET1":8279624.64669,"YAHOO_SET2":213.408355719,"MQ2007-list":41954.1355149,"MQ2008-list":10895.0834207,"AMAZON_poly2":8.01441262934}
best_pwacc = {"MQ2007":0.703244, "MQ2008":0.827171, "MSLR":0.615967, "YAHOO_SET1": 0.686103, "YAHOO_SET2": 0.697388, "MQ2007-list":0.806789, "MQ2008-list":0.821121, "AMAZON_poly2":0.873384}

train_exe = "solver/train"
time_limit = 7200
