# arquivo com configurações de teste para regressão logistica
# estamos utiizando o dataset Breast Cancer (cancer de mama) do scikit

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from regressao_logistica import RegressaoLogistica

# divir entre treino e teste
def dividir_dados(X, y, proporcao_treino=0.8, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    
    split = int(proporcao_treino * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalizar_dados(X_train, X_test):
    """
    Normaliza os dados (importante para gradiente descendente)
    """
    media = X_train.mean(axis=0)
    desvio = X_train.std(axis=0)
    
    X_train_norm = (X_train - media) / (desvio + 1e-8)
    X_test_norm = (X_test - media) / (desvio + 1e-8)
    
    return X_train_norm, X_test_norm


def main():
    print("=" * 60)
    print("REGRESSÃO LOGÍSTICA - Teste com Breast Cancer Dataset")
    print("=" * 60)
    
    # carregar dataset
    print("\n1. Carregando dados...")
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    print(f"   - Total de amostras: {len(X)}")
    print(f"   - Features: {len(cancer.feature_names)}")
    print(f"   - Classes: {cancer.target_names}")
    print(f"   - Distribuição: {np.sum(y == 0)} malignos, {np.sum(y == 1)} benignos")
    
    # dividir em treino e teste
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    print(f"\n2. Divisão dos dados:")
    print(f"   - Treino: {len(X_train)} amostras")
    print(f"   - Teste: {len(X_test)} amostras")
    
    # normalizar dados (importante!)
    print(f"\n3. Normalizando dados...")
    X_train_norm, X_test_norm = normalizar_dados(X_train, X_test)
    print(f"   - Dados normalizados (média 0, desvio 1)")
    
    # treinar o modelo
    print(f"\n4. Treinando modelo...")
    print(f"   - Learning rate: 0.1")
    print(f"   - Iterações: 1000")
    
    modelo = RegressaoLogistica(learning_rate=0.1, n_iteracoes=1000)
    modelo.fit(X_train_norm, y_train)
    
    # fazer predições
    print(f"\n5. Fazendo predições...")
    y_pred_train = modelo.predict(X_train_norm)
    y_pred_test = modelo.predict(X_test_norm)
    
    # calcular acurácias
    acuracia_train = modelo.score(X_train_norm, y_train)
    acuracia_test = modelo.score(X_test_norm, y_test)
    
    print(f"\n6. Resultados:")
    print(f"   - Acurácia Treino: {acuracia_train:.2%}")
    print(f"   - Acurácia Teste: {acuracia_test:.2%}")
    
    # matriz de confusão manual
    print(f"\n7. Matriz de Confusão (Teste):")
    tp = np.sum((y_test == 1) & (y_pred_test == 1))  # Verdadeiro Positivo
    tn = np.sum((y_test == 0) & (y_pred_test == 0))  # Verdadeiro Negativo
    fp = np.sum((y_test == 0) & (y_pred_test == 1))  # Falso Positivo
    fn = np.sum((y_test == 1) & (y_pred_test == 0))  # Falso Negativo
    
    print(f"                  Predito")
    print(f"                0       1")
    print(f"   Real    0   {tn:3}    {fp:3}")
    print(f"           1   {fn:3}    {tp:3}")
    
    # métricas adicionais
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
    
    print(f"\n8. Métricas:")
    print(f"   - Precisão: {precisao:.2%}")
    print(f"   - Recall: {recall:.2%}")
    print(f"   - F1-Score: {f1:.2%}")
    
    # exemplos de predições com probabilidades
    print(f"\n9. Exemplos de predições (com probabilidades):")
    probs_test = modelo.predict_proba(X_test_norm)
    for i in range(5):
        classe_real = cancer.target_names[y_test[i]]
        classe_pred = cancer.target_names[y_pred_test[i]]
        prob = probs_test[i]
        print(f"   Real: {classe_real:10} | Predito: {classe_pred:10} | "
              f"Prob: {prob:.4f}")
    
    # plotar curva de custo
    print(f"\n10. Gerando gráficos...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, modelo.n_iteracoes, 100), modelo.historico_custo, 
             linewidth=2, marker='o')
    plt.xlabel('Iteração', fontsize=12)
    plt.ylabel('Custo (Log Loss)', fontsize=12)
    plt.title('Convergência do Gradiente Descendente', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('convergencia.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico 1 salvo em: convergencia.png")
    plt.show()
    
    # distribuição de probabilidades
    probs_test = modelo.predict_proba(X_test_norm)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(probs_test[y_test == 0], bins=30, alpha=0.7, label='Maligno (0)', 
             color='red', edgecolor='black')
    plt.hist(probs_test[y_test == 1], bins=30, alpha=0.7, label='Benigno (1)', 
             color='blue', edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Probabilidade Predita', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.title('Distribuição das Probabilidades', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # plotar probabilidades ordenadas
    indices_ordenados = np.argsort(probs_test)
    cores = ['red' if y == 0 else 'blue' for y in y_test[indices_ordenados]]
    plt.scatter(range(len(probs_test)), probs_test[indices_ordenados], 
                c=cores, alpha=0.6, s=20)
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Amostra (ordenada)', fontsize=12)
    plt.ylabel('Probabilidade Predita', fontsize=12)
    plt.title('Probabilidades Ordenadas', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probabilidades.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico 2 salvo em: probabilidades.png")
    plt.show()
    
    # testar diferentes learning rates
    print(f"\n11. Testando diferentes learning rates...")
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    plt.figure(figsize=(12, 8))
    
    for lr in learning_rates:
        modelo_temp = RegressaoLogistica(learning_rate=lr, n_iteracoes=1000)
        modelo_temp.fit(X_train_norm, y_train)
        plt.plot(range(0, modelo_temp.n_iteracoes, 100), 
                modelo_temp.historico_custo, 
                linewidth=2, marker='o', label=f'LR = {lr}')
    
    plt.xlabel('Iteração', fontsize=12)
    plt.ylabel('Custo (Log Loss)', fontsize=12)
    plt.title('Convergência com Diferentes Learning Rates', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('diferentes_lr.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico 3 salvo em: diferentes_lr.png")
    plt.show()
    
    # visualização 2D (usando PCA manual simples - 2 primeiras features)
    print(f"\n12. Visualizando decision boundary (2D)...")
    X_train_2d = X_train_norm[:, :2]
    X_test_2d = X_test_norm[:, :2]
    
    modelo_2d = RegressaoLogistica(learning_rate=0.1, n_iteracoes=1000)
    modelo_2d.fit(X_train_2d, y_train)
    
    # criar grid para boundary
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = modelo_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
    plt.colorbar(label='Probabilidade Classe 1')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, 
               cmap='RdBu', s=50, edgecolors='black', linewidths=1.5)
    plt.xlabel(f'{cancer.feature_names[0]}', fontsize=12)
    plt.ylabel(f'{cancer.feature_names[1]}', fontsize=12)
    plt.title('Decision Boundary (2 features)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('decision_boundary.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico 4 salvo em: decision_boundary.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)


if __name__ == "__main__":
    main()
