
import config
import numpy as np
import pandas as pd

from benchmarks.pad20plus.dataset import PAD20Plus
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    print("- Loading the dataset")
    df = pd.read_csv(config.PAD_20_PLUS_RAW_METADATA)

    # cluster triage labels
    df[PAD20Plus.TARGET_COLUMN] = df[PAD20Plus.TARGET_COLUMN].map({
        "C43": "P1",
        "D03": "P1",
        "D22": "P1",

        "C80": "P2",
        "C44": "P2",
        "D04": "P2",
        "L75": "P2",  
        "D23": "P2",    

        "L57": "P3",
        "L25": "P3",
        "L30": "P3",
        "L98": "P3",    

        "L78": "P4",
        "L82": "P4",

        "L70": "P5",
        "00": "P5"
    })
    df = df.dropna(subset=[PAD20Plus.TARGET_COLUMN])

    df = df[df[PAD20Plus.IMAGE_SOURCE_COLUMN] == "CLINICAL"]
    df[PAD20Plus.IMAGE_COLUMN] = df[PAD20Plus.IMAGE_COLUMN].apply(lambda img : f'{img}.png')
    
    df.loc[:, PAD20Plus.NUMERICAL_FEATURES] = df[PAD20Plus.NUMERICAL_FEATURES].fillna(-1).astype(np.float32)

    df = df.reset_index(drop=True)

    print("- Splitting the dataset")

    # create cross-validation splits, grouping by patient and while stratifying by diagnostic
    kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    df['folder'] = None
    for i, (_, test_indexes) in enumerate(kfold.split(df, df[PAD20Plus.TARGET_COLUMN], groups=df[PAD20Plus.PATIENT_ID])):
        df.loc[test_indexes, 'folder'] = i + 1

    # Validate patient id separation across folders
    patient_ids = df.groupby('folder')[PAD20Plus.PATIENT_ID].unique()
    for i, ids in enumerate(patient_ids):
        for j, other_ids in enumerate(patient_ids):
            if i !=j and set(ids).intersection(other_ids):
                raise ValueError(f"Patient IDs {ids} and {other_ids} are present in the same folder {i+1} and {j+1}.")

    print("- Converting the labels to numbers")
    label_encoder = LabelEncoder()
    df[PAD20Plus.TARGET_COLUMN] = df[PAD20Plus.TARGET_COLUMN].astype('category')
    df[PAD20Plus.TARGET_NUMBER_COLUMN] = label_encoder.fit_transform(df[PAD20Plus.TARGET_COLUMN])

    # fix empty values
    df = df.replace(" ", np.nan).replace("  ", np.nan)

    # fix brazilian background
    df.loc[:, PAD20Plus.NUMERICAL_FEATURES] = df.loc[:, PAD20Plus.NUMERICAL_FEATURES].fillna(0).astype(np.float32)

    df = pd.get_dummies(df, columns=PAD20Plus.RAW_CATEGORICAL_FEATURES, dtype=np.int8)

    # create one-hot-encoded metadata parent folder
    config.DATA_PATH.mkdir(exist_ok=True)
    config.PAD_20_PLUS_ONE_HOT_ENCODED.parent.mkdir(exist_ok=True)

    # save one-hot-encoded metadata
    df.to_csv(config.PAD_20_PLUS_ONE_HOT_ENCODED, index=False)

    print("- Checking the target distribution")
    print(df[PAD20Plus.TARGET_COLUMN].value_counts())
    print(f"Total number of samples: {df[PAD20Plus.TARGET_COLUMN].value_counts().sum()}")