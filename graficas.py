import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

# Cargar datasets
haberman_data = pd.read_csv("c:/Users/Alejandro/Documents/datasets csv/haberman.csv")
car_data = pd.read_csv("c:/Users/Alejandro/Documents/datasets csv/car_evaluation.csv")

# Separar características y etiquetas
X_haberman, y_haberman = haberman_data.drop("survival_status", axis=1), haberman_data["survival_status"]
X_car, y_car = car_data.drop("class", axis=1), car_data["class"]

# Función para Hold-Out Validation
def hold_out_plot(X, y, dataset_name, test_size):
    X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_state=42)
    sizes = [len(X_train), len(X_test)]
    labels = ['Entrenamiento', 'Prueba']
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'Hold-Out ({dataset_name}) - Test Size {test_size}')
    plt.savefig(f"{dataset_name}_holdout.png")
    plt.show()

# Función para K-Fold Cross-Validation
def k_fold_plot(X, y, dataset_name, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_sizes = []
    test_sizes = []
    
    for train_index, test_index in kf.split(X):
        train_sizes.append(len(train_index))
        test_sizes.append(len(test_index))
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, k+1), train_sizes, label='Entrenamiento', marker='o')
    plt.plot(range(1, k+1), test_sizes, label='Prueba', marker='o')
    plt.title(f'K-Fold Validation ({dataset_name}) - K={k}')
    plt.xlabel('Fold')
    plt.ylabel('Tamaño del Conjunto')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dataset_name}_kfold.png")
    plt.show()

# Generar gráficos para ambos datasets y métodos
hold_out_plot(X_haberman, y_haberman, 'Haberman', test_size=0.3)
hold_out_plot(X_car, y_car, 'Car Evaluation', test_size=0.2)

k_fold_plot(X_haberman, y_haberman, 'Haberman', k=5)
k_fold_plot(X_car, y_car, 'Car Evaluation', k=4)