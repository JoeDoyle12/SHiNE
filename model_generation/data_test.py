import pickle

file_path = '../data_generation/TTV_files_ttv/final_data.pkl'

with open(file_path, 'rb') as pkl_file:
    data = pickle.load(pkl_file)
print([len(a) for a in data])