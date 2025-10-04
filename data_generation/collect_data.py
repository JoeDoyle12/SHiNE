import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os.path
from os import path
import json
import pickle
from utils import get_ttv_duration
import pprint
from zipfile import ZipFile
import io
import glob 
import sys 
import re
import math


def normalize(list):
    arr = np.asarray(list)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) 
    # return (arr - np.mean(arr)) / np.std(arr)

cwd = os.getcwd()
cwd = str(cwd)
Folder_name = "TTV_TDV_Data"

if not path.exists(cwd + '/TTV_files_ttv'):
    os.mkdir(cwd + '/TTV_files_ttv')

plot_path =  cwd + '/TTV_files_ttv'


parameter_names = ['m_t', 'm_nont', 
                   'e_t', 'e_nont', 
                   'incli_t', 'incli_nont', 
                   'a_t', 'a_nont', 
                   'omega_t', 'omega_nont',
                   'Omega_t', 'Omega_nont', 
                   'Mean_ano_t', 'Mean_ano_nont']


count = 0
par_count = 0 

total_dic = {}

files = glob.glob(cwd + "/sim_folder_"+ Folder_name +"/num_*")

high = []

with open(cwd + "/sim_folder_" + Folder_name + "/num_0_xPos/file_names.txt", 'r') as file:
    for line in file:
        print(line)
        high.append(float(float(re.search(r'inont_([0-9]+(?:\.[0-9]+)?)', line.strip()).group(1)) < (10 * math.pi / 180)))
print("High Inclination: ", high)

data = []
labels_to_output = []

for i in range(len(files)):
    file = files[i]
    cur_parent_path = file

    if not path.exists(cur_parent_path):
        print(cur_parent_path + ' not exists!')
        exit()

    print(f'currently reading {file}')

    with open(cur_parent_path + '/file_names.txt', "r") as file_names: # open file_names in num_* folder

        with ZipFile(cur_parent_path + '/reuslt.zip', 'r') as zip:
            for j, name in enumerate(file_names): # iterate Test
                
                
                name = name.split() # num_*Test_*, *file name
                
                TTV_file_name = name[0] + '.txt' # e.g., num_*_Test_*.txt

                ''' 
                1. check if the file name is valided/completed, and the file is in [*, 3] shape,
                ''' 
                try:
                    with io.BufferedReader(zip.open(TTV_file_name, mode='r')) as TTV_file:
                        transit_data = np.genfromtxt(TTV_file, dtype='str').reshape(-1, 3) 
                except:
                    continue 
                    
                if len(transit_data) == 0:
                    print(TTV_file_name + ' not started')
                    continue
                if len(transit_data) < 135*3:
                    print(TTV_file_name + ' not filled')
                     

                Test_dic = {}
                
                labels = name[1].split('_')
                flag = 1
                labels_number = []
                for label in labels:
                    if flag == 1:
                        flag *= -1
                        continue 
                    else:
                        labels_number.append(float(label))
                        flag *= -1
                parameters_dic = {}
                for k, para in enumerate(labels_number):
                    parameters_dic[parameter_names[k]] = para
                
                TTV_list, duration_list, \
                transit_list, fitted_Period_days, \
                percent_tdv, tdv_list = get_ttv_duration(transit_data)


                ttvs, tdvs = normalize(TTV_list), normalize(percent_tdv)

                pairs = [[ttvs[i], tdvs[i]] for i in range(len(ttvs))]

                data.append(pairs)
                labels_to_output.append(high[i])

                # average_dur_temp = np.mean(duration_list)
                # per_tdv_temp = (duration_list - average_dur_temp)/average_dur_temp

                # max_ttv = np.max(np.abs(TTV_list))
                # max_tdv = np.max(np.abs(per_tdv_temp))
                # ttv_amplitude = (np.max(TTV_list) - np.min(TTV_list)) * 24 *60 *60
                # tdv_amplitude = np.max(per_tdv_temp) - np.min(per_tdv_temp)
                
                # parameters_dic['P_t'] = fitted_Period_days

                # Test_dic['TTV_list'] = TTV_list # wrt 134
                # Test_dic['duration_list'] = duration_list
                # Test_dic['transit_list'] = transit_list 
                # Test_dic['parameters'] = parameters_dic 
                # Test_dic['percent_tdv'] = percent_tdv
                # Test_dic['name'] = name[0]
                # count += 1

                # print(Test_dic['TTV_list'], Test_dic['percent_tdv'])

with open(cwd + '/TTV_files_ttv/final_data.pkl', 'ab') as pkl_file:
    pickle.dump(data, pkl_file)

with open(cwd + '/TTV_files_ttv/final_labels.pkl', 'ab') as pkl_file:
    pickle.dump(labels_to_output, pkl_file)

print(len(files), count)

