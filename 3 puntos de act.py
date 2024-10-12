import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

# Cargar los datasets proporcionados
haberman_data = pd.read_csv("c:/Users/Alejandro/Documents/datasets csv/haberman.csv")
car_data = pd.read_csv("c:/Users/Alejandro/Documents/datasets csv/car_evaluation.csv")

# Separar características y etiquetas para ambos datasets
X_haberman = haberman_data.drop("survival_status", axis=1)
y_haberman = haberman_data["survival_status"]

X_car = car_data.drop("class", axis=1)
y_car = car_data["class"]

# Funciones de validación

def hold_out_validation(X, y, test_size):
    """Hold-Out Validation con tamaño de prueba definido por el usuario"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"Hold-Out - Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def k_fold_validation(X, y, k):
    """K-Fold Cross-Validation con K fijado por el usuario"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"K-Fold {i} - Train: {len(train_index)}, Test: {len(test_index)}")

def leave_one_out_validation(X, y):
    """Leave-One-Out Validation"""
    loo = LeaveOneOut()
    for i, (train_index, test_index) in enumerate(loo.split(X), 1):
        if i <= 5:  # Mostrar solo los primeros 5 splits para ilustración
            print(f"Leave-One-Out {i} - Train: {len(train_index)}, Test: {len(test_index)}")
        if i > 5:
            break

# Entrada del usuario para los parámetros
print("==== Haberman's Survival ====")
r = float(input("Ingresa el valor de r (proporción de prueba) para Hold-Out: "))
k = int(input("Ingresa el valor de K para K-Fold Cross-Validation: "))

print("\nHold-Out Validation:")
hold_out_validation(X_haberman, y_haberman, test_size=r)

print("\nK-Fold Validation:")
k_fold_validation(X_haberman, y_haberman, k=k)

print("\nLeave-One-Out Validation:")
leave_one_out_validation(X_haberman, y_haberman)

print("\n==== Car Evaluation ====")
r = float(input("Ingresa el valor de r (proporción de prueba) para Hold-Out: "))
k = int(input("Ingresa el valor de K para K-Fold Cross-Validation: "))

print("\nHold-Out Validation:")
hold_out_validation(X_car, y_car, test_size=r)

print("\nK-Fold Validation:")
k_fold_validation(X_car, y_car, k=k)

print("\nLeave-One-Out Validation:")
leave_one_out_validation(X_car, y_car)