import pickle
import glob

def get_data():
    data_files = sorted(glob.glob('data/final_data.pkl'))
    labels_files = sorted(glob.glob('data/final_labels.pkl'))

    data = []

    labels = []

    for i in range(len(data_files)):
        with open(data_files[i], 'rb') as pkl_file:
            internal_data = pickle.load(pkl_file)
        with open(labels_files[i], 'rb') as pkl_file:
            internal_labels = pickle.load(pkl_file)
        for j in range(len(internal_data)):
            if len(internal_data[j]) < 135:
                continue
            data.append(internal_data[j])
            labels.append(internal_labels[j])
    
    return data, labels

data, labels = get_data()
print(len(data))
print([len(a) for a in data], labels)