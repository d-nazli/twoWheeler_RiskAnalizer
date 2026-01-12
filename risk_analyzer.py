import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict
import numpy as np


def compute_class_probs(X_train, y_train):
    class_probs = {}
    feature_probs = defaultdict(dict)
    classes = y_train.unique()
    
    for cls in classes:
        X_cls = X_train[y_train == cls]
        class_probs[cls] = len(X_cls) / len(X_train)
        
        for col in X_train.columns:
            value_counts = X_cls[col].value_counts(normalize=True)
            feature_probs[cls][col] = value_counts.to_dict()
    
    return class_probs, feature_probs

def predict(X, class_probs, feature_probs):
    predictions = []
    for _, row in X.iterrows():
        scores = {}
        for cls in class_probs:
            prob = np.log(class_probs[cls])
            for col in X.columns:
                value = row[col]
                prob += np.log(feature_probs[cls][col].get(value, 1e-6))  # Laplace smoothing gibi
            scores[cls] = prob
        predicted = max(scores, key=scores.get)
        predictions.append(predicted)
    return predictions



def predict_single(new_data_row, class_probs, feature_probs):
    scores = {}
    for cls in class_probs:
        prob = np.log(class_probs[cls])
        for col, value in new_data_row.items():
            prob += np.log(feature_probs[cls][col].get(value, 1e-6))
        scores[cls] = prob
    predicted = max(scores, key=scores.get)
    return predicted



def ALLANALIZER(analizer):
    df = pd.read_csv("Bayes_Risk_Analiziornek_Veri_Seti.csv")

    df = df[['motor_uzaklık', 'motor_yön', 'motor_yönelme', 'motor_hızı', 'gerçek_risk']]

    category_orders = {
        'motor_uzaklık': ['yakın', 'orta', 'uzak'],
        'motor_yön': ['sol', 'ortada', 'sağ'],
        'motor_yönelme': ['yaklaşıyor', 'sabit', 'uzaklaşıyor'],
        'motor_hızı': ['düşük', 'sabit', 'yüksek'],
        'gerçek_risk': ['düşük', 'orta', 'yüksek']
    }

    for col in df.columns:
        df[col] = pd.Categorical(df[col], categories=category_orders[col], ordered=True).codes
        
    X = df.drop('gerçek_risk', axis=1)
    y = df['gerçek_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    inverse_category_orders = {col: {i: v for i, v in enumerate(category_orders[col])} for col in category_orders}

    new_sample = analizer

    encoded_sample = {}
    for col, val in new_sample.items():
        encoded_sample[col] = category_orders[col].index(val)



    class_probs, feature_probs = compute_class_probs(X_train, y_train)

    y_pred = predict(X_test, class_probs, feature_probs)
    predicted_label_code = predict_single(encoded_sample, class_probs, feature_probs)
    predicted_label_name = inverse_category_orders['gerçek_risk'][predicted_label_code]

    print("\nYeni motor verisi için tahmin edilen risk seviyesi:", predicted_label_name)
    return predicted_label_name

    


