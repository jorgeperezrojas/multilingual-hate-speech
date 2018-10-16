from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import hashlib

def hash_texto(texto):
    out = hashlib.sha224(texto.encode('utf-8')).hexdigest()
    return str(out)

def split_data_from_description(data_x, data_y, scenario, split_size, random_state):
    (source_sets, target_set) = scenario

    if target_set in source_sets:
        x_train, x_test, y_train, y_test = train_test_split(
            data_x[target_set], data_y[target_set], 
            test_size=split_size, random_state=random_state, stratify=data_y[target_set])
        source_sets.remove(target_set)
        list_of_x_train_data = [x_train] + [data_x[label] for label in source_sets]
        list_of_y_train_data = [y_train] + [data_y[label] for label in source_sets]
    else:
        list_of_x_train_data = [data_x[label] for label in source_sets]
        list_of_y_train_data = [data_y[label] for label in source_sets]
        if target_set != '':
            x_test = data_x[target_set]
            y_test = data_y[target_set]
    
    x_train = np.concatenate(list_of_x_train_data, axis=0)
    y_train = np.concatenate(list_of_y_train_data, axis=0)

    if target_set == '':
        x_train, x_test, y_train, y_test = train_test_split(
            x_train, y_train, test_size=split_size, random_state=random_state, stratify=y_train)

    x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
    x_test, y_test = shuffle(x_test, y_test, random_state=random_state)

    return x_train, x_test, y_train, y_test