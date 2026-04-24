import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

WINDOW_SIZES = [1,5,10,20]

raw_df = pd.read_csv('raw.csv')
net_df = pd.read_csv('syn1.csv')
our_df = pd.read_csv('syn2.csv')

for df in [raw_df, net_df, our_df]:
    df['type'] = df['type'].replace('background', 'normal')
    df['type'] = df['type'].replace('blacklist', 'anomaly')

le = LabelEncoder()
raw_df['type_enc'] = le.fit_transform(raw_df['type'])

net_df['type_enc'] = le.transform(net_df['type'])
our_df['type_enc'] = le.transform(our_df['type'])

print("mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

def add_multi_scale_features(df, base_features=['byt', 'pkt'], window_sizes=[1]):
    df_feat = df.copy()
    for f in base_features:
        for w in window_sizes:
            df_feat[f'{f}_mean_{w}'] = df_feat[f].rolling(w, min_periods=1).mean()
            df_feat[f'{f}_std_{w}'] = df_feat[f].rolling(w, min_periods=1).std().fillna(0)
            df_feat[f'{f}_max_{w}'] = df_feat[f].rolling(w, min_periods=1).max()
            df_feat[f'{f}_min_{w}'] = df_feat[f].rolling(w, min_periods=1).min()
            df_feat[f'{f}_diff_{w}'] = df_feat[f].diff(w).fillna(0)
            df_feat[f'{f}_ratio_{w}'] = df_feat[f] / (df_feat[f].rolling(w, min_periods=1).mean() + 1e-6)
    return df_feat

raw_df = add_multi_scale_features(raw_df, window_sizes=WINDOW_SIZES)
net_df = add_multi_scale_features(net_df, window_sizes=WINDOW_SIZES)
our_df = add_multi_scale_features(our_df, window_sizes=WINDOW_SIZES)

features = [c for c in raw_df.columns if c not in ['type', 'type_enc']]

X_raw = raw_df[features]
y_raw = raw_df['type_enc']

X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

X_net = net_df[features]
y_net = net_df['type_enc']

X_train_net, _, y_train_net, _ = train_test_split(
    X_net, y_net, test_size=0.2, random_state=42, stratify=y_net
)

X_our = our_df[features]
y_our = our_df['type_enc']

X_train_our, _, y_train_our, _ = train_test_split(
    X_our, y_our, test_size=0.2, random_state=42, stratify=y_our
)

scaler = StandardScaler()
X_train_raw_scaled = scaler.fit_transform(X_train_raw)

X_train_net_scaled = scaler.transform(X_train_net)
X_train_our_scaled = scaler.transform(X_train_our)

X_test_scaled = scaler.transform(X_test)

models_dict = {
    'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=1),
    'DT': DecisionTreeClassifier(random_state=1, class_weight='balanced'),
    'RF': RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced'),
    'GB': GradientBoostingClassifier(n_estimators=200, random_state=1),
    'LR': LogisticRegression(max_iter=1000, random_state=1, class_weight='balanced')
}

datasets = {
    'REAL': (X_train_raw_scaled, y_train_raw),
    'NetShare': (X_train_net_scaled, y_train_net),
    'Ours': (X_train_our_scaled, y_train_our)
}

results = []

for dataset_name, (X_train_scaled, y_train) in datasets.items():
    for model_name, model in models_dict.items():

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)

        results.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Accuracy': acc
        })

results_df = pd.DataFrame(results)

print("\n===== reaults =====")
print(results_df)

final_results = []

for model in results_df['Model'].unique():

    df_m = results_df[results_df['Model'] == model]

    acc_real = df_m[df_m['Dataset'] == 'REAL']['Accuracy'].values[0]
    acc_net = df_m[df_m['Dataset'] == 'NetShare']['Accuracy'].values[0]
    acc_our = df_m[df_m['Dataset'] == 'Ours']['Accuracy'].values[0]

    final_results.append({
        'Model': model,
        'REAL': acc_real,
        'NetShare': acc_net,
        'Ours': acc_our,
        'Gap(NetShare)': acc_real - acc_net,
        'Gap(Ours)': acc_real - acc_our
    })

final_df = pd.DataFrame(final_results)

print("\n===== results =====")
print(final_df)