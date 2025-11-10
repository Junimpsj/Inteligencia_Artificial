# arquivo com configurações de teste para o knn classficador
# estamos utiizando o dataset Iris (de flores) do scikit

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from knn_classificador import KNNClassificador

# treino e teste
def dividir_dados(X, y, proporcao_treino=0.8, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    
    split = int(proporcao_treino * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    print("=" * 60)
    print("KNN CLASSIFICADOR - Teste com Dataset Iris")
    print("=" * 60)
    
    # carregar dataset Iris
    print("\n1. Carregando dados...")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    print(f"   - Total de amostras: {len(X)}")
    print(f"   - Features: {iris.feature_names}")
    print(f"   - Classes: {iris.target_names}")
    
    # dividir em treino e teste
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    print(f"\n2. Divisão dos dados:")
    print(f"   - Treino: {len(X_train)} amostras")
    print(f"   - Teste: {len(X_test)} amostras")
    
    # treinar o modelo
    print(f"\n3. Treinando modelo com k=5...")
    knn = KNNClassificador(k=5)
    knn.fit(X_train, y_train)
    
    # fazer predições
    print(f"\n4. Fazendo predições...")
    y_pred = knn.predict(X_test)
    
    # calcular acurácia
    acuracia = knn.score(X_test, y_test)
    print(f"\n5. Resultado:")
    print(f"   - Acurácia: {acuracia:.2%}")
    
    # mostrar alguns exemplos de predições
    print(f"\n6. Exemplos de predições:")
    for i in range(5):
        print(f"   Real: {iris.target_names[y_test[i]]:15} | "
              f"Predito: {iris.target_names[y_pred[i]]}")
    
    # testar diferentes valores de K
    print(f"\n7. Testando diferentes valores de K...")
    k_valores = range(1, 21)
    acuracias = []
    
    for k in k_valores:
        knn_temp = KNNClassificador(k=k)
        knn_temp.fit(X_train, y_train)
        acc = knn_temp.score(X_test, y_test)
        acuracias.append(acc)
    
    melhor_k = k_valores[np.argmax(acuracias)]
    melhor_acuracia = max(acuracias)
    print(f"   - Melhor K: {melhor_k}")
    print(f"   - Melhor acurácia: {melhor_acuracia:.2%}")
    
    # plotar gráfico
    print(f"\n8. Gerando gráfico...")
    plt.figure(figsize=(10, 6))
    plt.plot(k_valores, acuracias, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Valor de K', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.title('Acurácia do KNN vs Valor de K', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_valores)
    plt.savefig('knn_acuracia.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico salvo em: knn_acuracia.png")
    plt.show()
    
    # visualizar predições (usando 2 features)
    print(f"\n9. Visualizando resultados (2D)...")
    X_train_2d = X_train[:, :2]  # Primeiras 2 features
    X_test_2d = X_test[:, :2]
    
    knn_2d = KNNClassificador(k=5)
    knn_2d.fit(X_train_2d, y_train)
    y_pred_2d = knn_2d.predict(X_test_2d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # dados de treino
    scatter1 = ax1.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train,
                          cmap='viridis', s=50, alpha=0.6, edgecolors='black')
    ax1.set_xlabel(iris.feature_names[0], fontsize=12)
    ax1.set_ylabel(iris.feature_names[1], fontsize=12)
    ax1.set_title('Dados de Treino', fontsize=14)
    plt.colorbar(scatter1, ax=ax1, label='Classe')
    
    # predições no teste
    scatter2 = ax2.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred_2d,
                          cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    # marcar erros com X vermelho
    erros = y_pred_2d != y_test
    if np.any(erros):
        ax2.scatter(X_test_2d[erros, 0], X_test_2d[erros, 1],
                   marker='x', s=200, c='red', linewidths=3, label='Erro')
        ax2.legend()
    ax2.set_xlabel(iris.feature_names[0], fontsize=12)
    ax2.set_ylabel(iris.feature_names[1], fontsize=12)
    ax2.set_title('Predições no Teste', fontsize=14)
    plt.colorbar(scatter2, ax=ax2, label='Classe Predita')
    
    plt.tight_layout()
    plt.savefig('knn_visualizacao.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico salvo em: knn_visualizacao.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)


if __name__ == "__main__":
    main()
