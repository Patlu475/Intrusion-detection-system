"""NSL-KDD and UNSW-NB15 data loading, preprocessing, and feature engineering."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
import torch

FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

COLUMN_NAMES = FEATURE_NAMES + ['attack_type', 'difficulty']

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

ATTACK_MAP = {
    'normal': 'Normal',
    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS',
    'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L',
    'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R',
}


def load_nsl_kdd(train_path, test_path):
    df_train = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    df_train.drop(columns=['difficulty'], inplace=True)
    df_test.drop(columns=['difficulty'], inplace=True)
    return df_train, df_test


def map_attacks_to_classes(df):
    df = df.copy()
    df['label'] = df['attack_type'].map(ATTACK_MAP)
    unmapped = df['label'].isna()
    if unmapped.any():
        unknown_attacks = df.loc[unmapped, 'attack_type'].unique()
        print(f"Warning: unmapped attacks {unknown_attacks}, assigning to 'Normal'")
        df.loc[unmapped, 'label'] = 'Normal'
    label_to_int = {name: i for i, name in enumerate(CLASS_NAMES)}
    df['label_int'] = df['label'].map(label_to_int)
    df.drop(columns=['attack_type', 'label'], inplace=True)
    return df


def encode_features(df_train, df_test):
    combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    for col in CATEGORICAL_COLS:
        combined[col] = pd.Categorical(combined[col])

    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLS, dtype=float)

    n_train = len(df_train)
    df_train_enc = combined.iloc[:n_train].reset_index(drop=True)
    df_test_enc = combined.iloc[n_train:].reset_index(drop=True)

    feature_cols = [c for c in df_train_enc.columns if c != 'label_int']
    X_train = df_train_enc[feature_cols].values.astype(np.float32)
    X_test = df_test_enc[feature_cols].values.astype(np.float32)
    y_train = df_train_enc['label_int'].values.astype(np.int64)
    y_test = df_test_enc['label_int'].values.astype(np.int64)

    return X_train, X_test, y_train, y_test, feature_cols


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train.astype(np.float32), X_test.astype(np.float32), scaler


def apply_smote(X_train, y_train):
    for k in [5, 3, 1]:
        try:
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            print(f"SMOTE applied (k_neighbors={k}). "
                  f"Train size: {len(X_train)} -> {len(X_res)}")
            return X_res.astype(np.float32), y_res.astype(np.int64)
        except ValueError:
            print(f"SMOTE k_neighbors={k} failed, trying lower...")

    print("SMOTE failed entirely, returning original data.")
    return X_train, y_train


def get_data(data_dir='data', use_smote=True):
    import os
    train_path = os.path.join(data_dir, 'KDDTrain+.txt')
    test_path = os.path.join(data_dir, 'KDDTest+.txt')

    print("Loading NSL-KDD dataset...")
    df_train, df_test = load_nsl_kdd(train_path, test_path)

    print("Mapping attacks to 5 classes...")
    df_train = map_attacks_to_classes(df_train)
    df_test = map_attacks_to_classes(df_test)

    print("One-hot encoding categorical features...")
    X_train, X_test, y_train, y_test, feature_names = encode_features(df_train, df_test)
    print(f"Feature dimensions: {X_train.shape[1]}")

    original_class_counts = np.bincount(y_train, minlength=5)

    print("Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    if use_smote:
        print("Applying SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train)

    print(f"Final shapes — Train: {X_train.shape}, Test: {X_test.shape}")
    for i, name in enumerate(CLASS_NAMES):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"  {name}: train={train_count}, test={test_count}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'num_classes': len(CLASS_NAMES),
        'num_features': X_train.shape[1],
        'class_names': CLASS_NAMES,
        'feature_names': feature_names,
        'original_class_counts': original_class_counts,
    }


def get_dataloaders(X_train, y_train, X_test, y_test, batch_size=256):
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ============================================================
# UNSW-NB15 Dataset
# ============================================================

UNSW_CLASS_NAMES = [
    'Normal', 'Analysis', 'Backdoor', 'DoS', 'Exploits',
    'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms',
]

UNSW_DROP_COLS = ['id', 'label']

UNSW_CATEGORICAL_COLS = ['proto', 'service', 'state']


def load_unsw_nb15(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    for df in [df_train, df_test]:
        df.columns = df.columns.str.strip().str.lower()

    return df_train, df_test


def map_unsw_attacks(df):
    df = df.copy()
    df['attack_cat'] = df['attack_cat'].fillna('Normal').str.strip()
    df.loc[df['attack_cat'] == '', 'attack_cat'] = 'Normal'

    cat_map = {
        'Normal': 'Normal',
        'Analysis': 'Analysis',
        'Backdoor': 'Backdoor',
        'Backdoors': 'Backdoor',
        'DoS': 'DoS',
        'Exploits': 'Exploits',
        'Fuzzers': 'Fuzzers',
        'Generic': 'Generic',
        'Reconnaissance': 'Reconnaissance',
        'Shellcode': 'Shellcode',
        'Worms': 'Worms',
    }
    df['attack_cat'] = df['attack_cat'].map(cat_map)
    unmapped = df['attack_cat'].isna()
    if unmapped.any():
        print(f"Warning: unmapped UNSW categories, assigning to Normal")
        df.loc[unmapped, 'attack_cat'] = 'Normal'

    label_to_int = {name: i for i, name in enumerate(UNSW_CLASS_NAMES)}
    df['label_int'] = df['attack_cat'].map(label_to_int)
    df.drop(columns=['attack_cat'] + [c for c in UNSW_DROP_COLS if c in df.columns],
            inplace=True)
    return df


def encode_unsw_features(df_train, df_test):
    combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    for col in UNSW_CATEGORICAL_COLS:
        if col in combined.columns:
            combined[col] = combined[col].fillna('unknown').astype(str)
            combined[col] = pd.Categorical(combined[col])

    combined = pd.get_dummies(combined, columns=UNSW_CATEGORICAL_COLS, dtype=float)

    for col in combined.select_dtypes(include=['object']).columns:
        if col != 'label_int':
            combined[col] = pd.to_numeric(combined[col], errors='coerce').fillna(0)

    combined = combined.fillna(0)

    n_train = len(df_train)
    df_train_enc = combined.iloc[:n_train].reset_index(drop=True)
    df_test_enc = combined.iloc[n_train:].reset_index(drop=True)

    feature_cols = [c for c in df_train_enc.columns if c != 'label_int']
    X_train = df_train_enc[feature_cols].values.astype(np.float32)
    X_test = df_test_enc[feature_cols].values.astype(np.float32)
    y_train = df_train_enc['label_int'].values.astype(np.int64)
    y_test = df_test_enc['label_int'].values.astype(np.int64)

    return X_train, X_test, y_train, y_test, feature_cols


def get_unsw_data(data_dir='data', use_smote=True):
    import os
    train_path = os.path.join(data_dir, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(data_dir, 'UNSW_NB15_testing-set.csv')

    print("Loading UNSW-NB15 dataset...")
    df_train, df_test = load_unsw_nb15(train_path, test_path)

    print("Mapping attack categories to 10 classes...")
    df_train = map_unsw_attacks(df_train)
    df_test = map_unsw_attacks(df_test)

    print("One-hot encoding categorical features...")
    X_train, X_test, y_train, y_test, feature_names = encode_unsw_features(
        df_train, df_test,
    )
    print(f"Feature dimensions: {X_train.shape[1]}")

    num_classes = len(UNSW_CLASS_NAMES)
    original_class_counts = np.bincount(y_train, minlength=num_classes)

    print("Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    if use_smote:
        print("Applying SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train)

    print(f"Final shapes — Train: {X_train.shape}, Test: {X_test.shape}")
    for i, name in enumerate(UNSW_CLASS_NAMES):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"  {name}: train={train_count}, test={test_count}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'num_classes': num_classes,
        'num_features': X_train.shape[1],
        'class_names': UNSW_CLASS_NAMES,
        'feature_names': feature_names,
        'original_class_counts': original_class_counts,
    }
