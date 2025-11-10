# arquivo com configurações de teste para a decision tree
# estamos utiizando o dataset Iris (de flores) do scikit

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from decision_tree import DecisionTree

# dividindo dados para treino e para teste
def dividir_dados(X, y, proporcao_treino=0.8, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    
    split = int(proporcao_treino * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    print("=" * 60)
    print("DECISION TREE - Teste com Dataset Iris")
    print("=" * 60)
    
    # carregar dataset
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
    print(f"\n3. Treinando Decision Tree...")
    print(f"   - Profundidade máxima: 5")
    print(f"   - Mínimo de amostras para split: 2")
    
    tree = DecisionTree(max_profundidade=5, min_amostras_split=2)
    tree.fit(X_train, y_train)
    
    # fazer predições
    print(f"\n4. Fazendo predições...")
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)
    
    # calcular acurácias
    acuracia_train = tree.score(X_train, y_train)
    acuracia_test = tree.score(X_test, y_test)
    
    print(f"\n5. Resultados:")
    print(f"   - Acurácia Treino: {acuracia_train:.2%}")
    print(f"   - Acurácia Teste: {acuracia_test:.2%}")
    
    # mostrar alguns exemplos
    print(f"\n6. Exemplos de predições:")
    for i in range(5):
        print(f"   Real: {iris.target_names[y_test[i]]:15} | "
              f"Predito: {iris.target_names[y_pred_test[i]]}")
    
    # testar diferentes profundidades
    print(f"\n7. Testando diferentes profundidades...")
    profundidades = range(1, 11)
    acuracias_train = []
    acuracias_test = []
    
    for prof in profundidades:
        tree_temp = DecisionTree(max_profundidade=prof, min_amostras_split=2)
        tree_temp.fit(X_train, y_train)
        acc_train = tree_temp.score(X_train, y_train)
        acc_test = tree_temp.score(X_test, y_test)
        acuracias_train.append(acc_train)
        acuracias_test.append(acc_test)
    
    melhor_prof = profundidades[np.argmax(acuracias_test)]
    melhor_acc = max(acuracias_test)
    print(f"   - Melhor profundidade: {melhor_prof}")
    print(f"   - Melhor acurácia teste: {melhor_acc:.2%}")
    
    # plotar gráfico de profundidade
    print(f"\n8. Gerando gráficos...")
    plt.figure(figsize=(10, 6))
    plt.plot(profundidades, acuracias_train, marker='o', linewidth=2, 
             markersize=8, label='Treino')
    plt.plot(profundidades, acuracias_test, marker='s', linewidth=2, 
             markersize=8, label='Teste')
    plt.xlabel('Profundidade Máxima', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.title('Acurácia vs Profundidade da Árvore', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(profundidades)
    plt.savefig('profundidade.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico 1 salvo em: profundidade.png")
    plt.show()
    
    # visualizar decision boundary (2D)
    print(f"\n9. Visualizando decision boundaries (2D)...")
    X_2d = X[:, :2]  # Primeiras 2 features
    X_train_2d, X_test_2d, _, _ = dividir_dados(X_2d, y)
    
    tree_2d = DecisionTree(max_profundidade=5, min_amostras_split=2)
    tree_2d.fit(X_train_2d, y_train)
    
    # criar grid
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = tree_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test,
                         cmap='viridis', s=100, edgecolors='black', linewidths=1.5)
    plt.xlabel(iris.feature_names[0], fontsize=12)
    plt.ylabel(iris.feature_names[1], fontsize=12)
    plt.title('Decision Boundary - Decision Tree', fontsize=14)
    plt.colorbar(scatter, label='Classe')
    plt.grid(True, alpha=0.3)
    plt.savefig('decision_boundary.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico 2 salvo em: decision_boundary.png")
    plt.show()
    
    # comparar diferentes profundidades visualmente
    print(f"\n10. Comparando diferentes profundidades visualmente...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    profundidades_viz = [1, 3, 5, 10]
    
    for idx, prof in enumerate(profundidades_viz):
        tree_temp = DecisionTree(max_profundidade=prof, min_amostras_split=2)
        tree_temp.fit(X_train_2d, y_train)
        
        Z_temp = tree_temp.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_temp = Z_temp.reshape(xx.shape)
        
        axes[idx].contourf(xx, yy, Z_temp, alpha=0.4, cmap='viridis')
        axes[idx].scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test,
                         cmap='viridis', s=50, edgecolors='black', linewidths=1)
        
        acc_temp = tree_temp.score(X_test_2d, y_test)
        axes[idx].set_xlabel(iris.feature_names[0], fontsize=11)
        axes[idx].set_ylabel(iris.feature_names[1], fontsize=11)
        axes[idx].set_title(f'Profundidade = {prof} (Acc: {acc_temp:.2%})', 
                           fontsize=12)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacao_profundidades.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico 3 salvo em: comparacao_profundidades.png")
    plt.show()
    
    # matriz de confusão
    print(f"\n11. Matriz de Confusão (Teste):")
    n_classes = len(iris.target_names)
    matriz_confusao = np.zeros((n_classes, n_classes), dtype=int)
    
    for real, pred in zip(y_test, y_pred_test):
        matriz_confusao[real, pred] += 1
    
    print("\n                    Predito")
    print("               ", end="")
    for i in range(n_classes):
        print(f"{iris.target_names[i][:10]:>12}", end="")
    print()
    
    for i in range(n_classes):
        print(f"   {iris.target_names[i][:10]:>10}", end="")
        for j in range(n_classes):
            print(f"{matriz_confusao[i, j]:>12}", end="")
        print()
    
    # visualizar matriz de confusão
    plt.figure(figsize=(8, 6))
    plt.imshow(matriz_confusao, interpolation='nearest', cmap='Blues')
    plt.title('Matriz de Confusão', fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    
    # adicionar valores nas células
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, matriz_confusao[i, j],
                    ha="center", va="center",
                    color="white" if matriz_confusao[i, j] > matriz_confusao.max()/2 else "black")
    
    plt.ylabel('Classe Real', fontsize=12)
    plt.xlabel('Classe Predita', fontsize=12)
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=100, bbox_inches='tight')
    print(f"\n   - Gráfico 4 salvo em: matriz_confusao.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)


if __name__ == "__main__":
    main()
