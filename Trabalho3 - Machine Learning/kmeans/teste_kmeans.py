# arquivo com configurações de teste para o k-means
# estamos utiizando o dataset Iris (de flores) do scikit para visualação de cluster

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from kmeans import KMeans


def main():
    print("=" * 60)
    print("K-MEANS CLUSTERING - Teste com Dataset Iris")
    print("=" * 60)
    
    # carregando o dataset Iris
    print("\n1. Carregando dados...")
    iris = datasets.load_iris()
    X = iris.data
    y_real = iris.target  # Para comparação (não usado no treino!)
    
    print(f"   - Total de amostras: {len(X)}")
    print(f"   - Features: {iris.feature_names}")
    print(f"   - Classes reais: {iris.target_names}")
    
    # método do Cotovelo para encontrar K ideal
    print(f"\n2. Usando Método do Cotovelo para encontrar K ideal...")
    kmeans_teste = KMeans(k=3)  # k temporário
    inercias = kmeans_teste.metodo_cotovelo(X, k_max=10)
    
    # plotar método do cotovelo
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inercias, marker='o', linewidth=2, markersize=10)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Inércia (soma das distâncias ao quadrado)', fontsize=12)
    plt.title('Método do Cotovelo', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 11))
    plt.savefig('metodo_cotovelo.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico salvo em: metodo_cotovelo.png")
    print(f"   - Observe onde a curva faz um 'cotovelo' (diminui menos)")
    plt.show()
    
    # treinar com K=3 (sabemos que Iris tem 3 classes)
    print(f"\n3. Treinando K-Means com k=3...")
    kmeans = KMeans(k=3, max_iteracoes=100, seed=42)
    kmeans.fit(X)
    
    print(f"   - Convergência alcançada")
    print(f"   - Inércia final: {kmeans.inertia:.2f}")
    
    # mostrar centroides
    print(f"\n4. Centroides finais:")
    for i, centroide in enumerate(kmeans.centroides):
        print(f"   Cluster {i}: {centroide}")
    
    # contar pontos por cluster
    print(f"\n5. Distribuição dos pontos:")
    for i in range(kmeans.k):
        count = np.sum(kmeans.labels == i)
        print(f"   Cluster {i}: {count} pontos")
    
    # visualizar clusters (usando 2 features)
    print(f"\n6. Visualizando clusters (2D)...")
    X_2d = X[:, :2]  # Primeiras 2 features
    
    # treinar novamente com dados 2D para visualização
    kmeans_2d = KMeans(k=3, seed=42)
    kmeans_2d.fit(X_2d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # classes reais
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_real,
                          cmap='viridis', s=50, alpha=0.6, edgecolors='black')
    ax1.set_xlabel(iris.feature_names[0], fontsize=12)
    ax1.set_ylabel(iris.feature_names[1], fontsize=12)
    ax1.set_title('Classes Reais', fontsize=14)
    plt.colorbar(scatter1, ax=ax1, label='Classe')
    
    # clusters encontrados
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_2d.labels,
                          cmap='viridis', s=50, alpha=0.6, edgecolors='black')
    ax2.scatter(kmeans_2d.centroides[:, 0], kmeans_2d.centroides[:, 1],
               marker='X', s=300, c='red', edgecolors='black', 
               linewidths=2, label='Centroides')
    ax2.set_xlabel(iris.feature_names[0], fontsize=12)
    ax2.set_ylabel(iris.feature_names[1], fontsize=12)
    ax2.set_title('Clusters K-Means', fontsize=14)
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    plt.tight_layout()
    plt.savefig('kmeans_clusters.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico salvo em: kmeans_clusters.png")
    plt.show()
    
    # visualizar todas as combinações de features
    print(f"\n7. Visualizando múltiplas projeções...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for idx, (i, j) in enumerate(feature_pairs):
        X_pair = X[:, [i, j]]
        kmeans_pair = KMeans(k=3, seed=42)
        kmeans_pair.fit(X_pair)
        
        axes[idx].scatter(X_pair[:, 0], X_pair[:, 1], c=kmeans_pair.labels,
                         cmap='viridis', s=50, alpha=0.6, edgecolors='black')
        axes[idx].scatter(kmeans_pair.centroides[:, 0], kmeans_pair.centroides[:, 1],
                         marker='X', s=200, c='red', edgecolors='black', linewidths=2)
        axes[idx].set_xlabel(iris.feature_names[i], fontsize=10)
        axes[idx].set_ylabel(iris.feature_names[j], fontsize=10)
        axes[idx].set_title(f'{iris.feature_names[i][:15]} vs {iris.feature_names[j][:15]}', 
                           fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kmeans_multiplas_projecoes.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico salvo em: kmeans_multiplas_projecoes.png")
    plt.show()
    
    # comparar diferentes valores de K
    print(f"\n8. Comparando diferentes valores de K...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    k_valores = [2, 3, 4, 5]
    
    for idx, k in enumerate(k_valores):
        kmeans_temp = KMeans(k=k, seed=42)
        kmeans_temp.fit(X_2d)
        
        axes[idx].scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_temp.labels,
                         cmap='viridis', s=50, alpha=0.6, edgecolors='black')
        axes[idx].scatter(kmeans_temp.centroides[:, 0], kmeans_temp.centroides[:, 1],
                         marker='X', s=200, c='red', edgecolors='black', linewidths=2)
        axes[idx].set_xlabel(iris.feature_names[0], fontsize=11)
        axes[idx].set_ylabel(iris.feature_names[1], fontsize=11)
        axes[idx].set_title(f'K = {k} (Inércia: {kmeans_temp.inertia:.2f})', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kmeans_diferentes_k.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico salvo em: kmeans_diferentes_k.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)


if __name__ == "__main__":
    main()
