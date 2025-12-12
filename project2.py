# === IMPORTS ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\n===== CHARGEMENT DU DATASET =====")
df = sns.load_dataset('titanic')
print(df.head())
df.shape
print(df.shape)


# ===================================================================
print("\n===== NETTOYAGE DES DONNÉES =====")

print("\nValeurs manquantes avant nettoyage :")
print(df.isnull().sum())

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.drop(['deck', 'embark_town', 'alive', 'who'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)

print("\nValeurs manquantes après nettoyage :")
print(df.isnull().sum())

print("\nDimensions du dataset après nettoyage :", df.shape)

# ===================================================================
print("\n===== FEATURE ENGINEERING =====")

df['FamilySize'] = df['sibsp'] + df['parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print(df[['sibsp','parch','FamilySize','IsAlone']].head())

df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
         'embarked', 'FamilySize', 'IsAlone', 'survived']]

print("\nVariables finales utilisées :")
print(df.columns)

# ===================================================================
print("\n===== ENCODAGE =====")

# Sex -> numérique
df['sex'] = df['sex'].map({'male': 1, 'female': 0})

# Encodage one-hot
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

print("\nExtrait du dataset après encodage :")
print(df.head())

# ===================================================================
print("\n===== OUTLIERS SUR 'fare' =====")

Q1, Q3 = df['fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

print(f"Bornes : {lower:.2f}  ->  {upper:.2f}")
print("Taille avant filtre :", df.shape)

df = df[(df['fare'] >= lower) & (df['fare'] <= upper)]

print("Taille après filtre :", df.shape)

# Transformation log
df['fare'] = np.log1p(df['fare'])

print("\nAperçu après transformation log sur 'fare' :")
print(df['fare'].describe())

# ===================================================================
print("\n===== SCALING (Standardisation) =====")

X = df.drop('survived', axis=1)
y = df['survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Exemple de données normalisées :")
print(X_scaled[:5])

# ===================================================================
print("\n===== SPLIT TRAIN / TEST =====")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Taille du train :", X_train.shape)
print("Taille du test  :", X_test.shape)

# ===================================================================
print("\n===== RÉGRESSION LOGISTIQUE =====")

log_model = LogisticRegression(max_iter=300)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report :\n", classification_report(y_test, y_pred_log))

# ===================================================================
print("\n===== KNN =====")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report :\n", classification_report(y_test, y_pred_knn))

# ===================================================================
print("\n===== VISUALISATION DES PERFORMANCES =====")

models = ["Logistic Regression", "KNN"]
accuracies = [accuracy_score(y_test, y_pred_log),
              accuracy_score(y_test, y_pred_knn)]

plt.figure(figsize=(7,5))
bars = plt.bar(models, accuracies)
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{bar.get_height():.3f}", ha='center')

plt.show()

print("\n===== FIN DU PROJECT 2 =====")
