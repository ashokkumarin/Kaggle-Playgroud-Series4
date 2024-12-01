import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical(dataset, columns):
    for c in columns:
        lbl = LabelEncoder()
        lbl.fit(list(dataset[c].values))
        # print(set(dataset[c].values))
        dataset[c] = lbl.transform(list(dataset[c].values))
        # print(set(dataset[c].values))

    return dataset


def remove_outliers(dataset, outliers):
    for element in outliers:
        if outliers[element][1] == '>':
            dataset = dataset[dataset[element] > outliers[element][0]]
        else:
            dataset = dataset[dataset[element] < outliers[element][0]]

    return dataset


def fill_dummy_values(dataset):
    null_columns = dataset.columns[dataset.isnull().any()]

    for column in null_columns:
        if dataset[column].dtype == 'object':
            dataset[column] = dataset[column].fillna('None')
        else:
            dataset[column] = dataset[column].fillna(int(0)) # interpolate()  # fillna(int(0))


def encode(x, v): return 1 if x == v else 0


def dichotomous_encode(dataset, columns):
    for col in columns:
        if col not in dataset.columns:
            continue
        idx = pd.Index(dataset[col])
        max_element = idx.value_counts().index[0]

        for item in idx.value_counts().index:
            pd.set_option('future.no_silent_downcasting', True)
            dataset[col] = dataset[col].replace(item, encode(item, max_element))
            dataset[col] = dataset[col].infer_objects(copy=False)  # Explicitly call infer_objects

    return dataset

