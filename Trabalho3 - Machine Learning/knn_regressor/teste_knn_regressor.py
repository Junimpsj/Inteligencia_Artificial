# arquivo com configurações de teste para o knn regressor
# estamos utiizando o dataset California Housing (preços de casas) do scikit

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from knn_regressor import KNNRegressor

# deividindo treino e teste
def dividir_dados(X, y, proporcao_treino=0.8, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    
    split = int(proporcao_treino * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    print("=" * 60)
    print("KNN REGRESSOR - Teste com Dataset California Housing")
    print("=" * 60)
    
    # carregar dataset
    print("\n1. Carregando dados...")
    california = datasets.fetch_california_housing()
    
    # usar apenas uma amostra para velocidade
    np.random.seed(42)
    indices = np.random.permutation(len(california.data))[:2000]
    X = california.data[indices]
    y = california.target[indices]
    
    print(f"   - Total de amostras: {len(X)}")
    print(f"   - Features: {california.feature_names}")
    print(f"   - Target: Preço médio das casas (em $100,000)")
    
    # dividir em treino e teste
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    print(f"\n2. Divisão dos dados:")
    print(f"   - Treino: {len(X_train)} amostras")
    print(f"   - Teste: {len(X_test)} amostras")
    
    # treinar o modelo
    print(f"\n3. Treinando modelo com k=5...")
    knn = KNNRegressor(k=5)
    knn.fit(X_train, y_train)
    
    # fazer predições
    print(f"\n4. Fazendo predições...")
    y_pred = knn.predict(X_test)
    
    # calcular métricas
    r2 = knn.score(X_test, y_test)
    mse = knn.mean_squared_error(X_test, y_test)
    rmse = np.sqrt(mse)
    
    print(f"\n5. Resultados:")
    print(f"   - R² Score: {r2:.4f}")
    print(f"   - MSE: {mse:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - Interpretação: Erro médio de ${rmse * 100000:.2f} no preço")
    
    # mostrar alguns exemplos
    print(f"\n6. Exemplos de predições:")
    for i in range(5):
        print(f"   Real: ${y_test[i]*100000:>10.2f} | "
              f"Predito: ${y_pred[i]*100000:>10.2f} | "
              f"Erro: ${abs(y_test[i]-y_pred[i])*100000:>10.2f}")
    
    # testar diferentes valores de K
    print(f"\n7. Testando diferentes valores de K...")
    k_valores = range(1, 21)
    r2_scores = []
    mse_scores = []
    
    for k in k_valores:
        knn_temp = KNNRegressor(k=k)
        knn_temp.fit(X_train, y_train)
        r2 = knn_temp.score(X_test, y_test)
        mse = knn_temp.mean_squared_error(X_test, y_test)
        r2_scores.append(r2)
        mse_scores.append(mse)
    
    melhor_k = k_valores[np.argmax(r2_scores)]
    melhor_r2 = max(r2_scores)
    print(f"   - Melhor K: {melhor_k}")
    print(f"   - Melhor R²: {melhor_r2:.4f}")
    
    # plotar gráficos
    print(f"\n8. Gerando gráficos...")
    
    # gráfico 1: Valores reais vs preditos
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Predição Perfeita')
    plt.xlabel('Valor Real', fontsize=12)
    plt.ylabel('Valor Predito', fontsize=12)
    plt.title('Valores Reais vs Predições - KNN Regressor', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('knn_real_vs_pred.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico 1 salvo em: knn_real_vs_pred.png")
    plt.show()
    
    # gráfico 2: R² e MSE vs K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(k_valores, r2_scores, marker='o', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Valor de K', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² vs Valor de K', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_valores)
    
    ax2.plot(k_valores, mse_scores, marker='o', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Valor de K', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('MSE vs Valor de K', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_valores)
    
    plt.tight_layout()
    plt.savefig('knn_metricas.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico 2 salvo em: knn_metricas.png")
    plt.show()
    
    # gráfico 3: Distribuição dos erros
    erros = y_test - y_pred
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(erros, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Erro de Predição', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.title('Distribuição dos Erros', fontsize=14)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, erros, alpha=0.6, s=50, edgecolors='black')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Valores Preditos', fontsize=12)
    plt.ylabel('Resíduos (Real - Predito)', fontsize=12)
    plt.title('Análise de Resíduos', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('knn_erros.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico 3 salvo em: knn_erros.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)


if __name__ == "__main__":
    main()
