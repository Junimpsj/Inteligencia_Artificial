# arquivo com configurações de teste para rede neural
# estamos utiizando diferentes datasets do scikit

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from rede_neural import RedeNeural

# dividindo treino e teste
def dividir_dados(X, y, proporcao_treino=0.8, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    
    split = int(proporcao_treino * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# normalizando os dados
def normalizar_dados(X_train, X_test):
    media = X_train.mean(axis=0)
    desvio = X_train.std(axis=0)
    
    X_train_norm = (X_train - media) / (desvio + 1e-8)
    X_test_norm = (X_test - media) / (desvio + 1e-8)
    
    return X_train_norm, X_test_norm

# testando com dataset IRIS
def teste_iris():
    print("=" * 60)
    print("TESTE 1: IRIS DATASET (Classificação Multiclasse)")
    print("=" * 60)
    
    # carregar dados
    print("\n1. Carregando dados...")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    print(f"   - Amostras: {len(X)}")
    print(f"   - Features: {len(iris.feature_names)}")
    print(f"   - Classes: {len(iris.target_names)}")
    
    # dividir e normalizar
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    X_train_norm, X_test_norm = normalizar_dados(X_train, X_test)
    
    # treinar rede neural
    print(f"\n2. Treinando Rede Neural...")
    print(f"   - Arquitetura: [4, 8, 3]")
    print(f"   - Ativação: sigmoid")
    print(f"   - Learning rate: 0.1")
    
    rede = RedeNeural(camadas=[4, 8, 3], learning_rate=0.1, 
                     n_iteracoes=2000, ativacao='sigmoid')
    rede.fit(X_train_norm, y_train)
    
    # avaliar
    acc_train = rede.score(X_train_norm, y_train)
    acc_test = rede.score(X_test_norm, y_test)
    
    print(f"\n3. Resultados:")
    print(f"   - Acurácia Treino: {acc_train:.2%}")
    print(f"   - Acurácia Teste: {acc_test:.2%}")
    
    # exemplos de predições
    print(f"\n4. Exemplos de predições:")
    y_pred = rede.predict(X_test_norm)
    probs = rede.predict_proba(X_test_norm)
    
    for i in range(5):
        print(f"   Real: {iris.target_names[y_test[i]]:15} | "
              f"Predito: {iris.target_names[y_pred[i]]:15} | "
              f"Confiança: {probs[i, y_pred[i]]:.2%}")
    
    # plotar convergência
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, rede.n_iteracoes, 100), rede.historico_custo, 
             linewidth=2, marker='o')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Custo (Cross-Entropy)', fontsize=12)
    plt.title('Convergência - Iris Dataset', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('convergencia_iris.png', dpi=100, bbox_inches='tight')
    print(f"\n   - Gráfico salvo em: convergencia_iris.png")
    plt.show()
    
    return rede, X_test_norm, y_test, iris

# testando com dataset DIGITS
def teste_digits():
    print("\n\n" + "=" * 60)
    print("TESTE 2: DIGITS DATASET (Dígitos Manuscritos)")
    print("=" * 60)
    
    # carregar dados
    print("\n1. Carregando dados...")
    digits = datasets.load_digits()
    X = digits.data  # 64 features (8x8 pixels)
    y = digits.target  # 10 classes (0-9)
    
    print(f"   - Amostras: {len(X)}")
    print(f"   - Features: {X.shape[1]} (imagens 8x8)")
    print(f"   - Classes: 10 (dígitos 0-9)")
    
    # dividir e normalizar
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    X_train_norm, X_test_norm = normalizar_dados(X_train, X_test)
    
    # treinar rede neural com ReLU
    print(f"\n2. Treinando Rede Neural...")
    print(f"   - Arquitetura: [64, 32, 16, 10]")
    print(f"   - Ativação: ReLU")
    print(f"   - Learning rate: 0.01")
    
    rede = RedeNeural(camadas=[64, 32, 16, 10], learning_rate=0.01, 
                     n_iteracoes=2000, ativacao='relu')
    rede.fit(X_train_norm, y_train)
    
    # avaliar
    acc_train = rede.score(X_train_norm, y_train)
    acc_test = rede.score(X_test_norm, y_test)
    
    print(f"\n3. Resultados:")
    print(f"   - Acurácia Treino: {acc_train:.2%}")
    print(f"   - Acurácia Teste: {acc_test:.2%}")
    
    # mostrar exemplos de predições com imagens
    print(f"\n4. Visualizando predições...")
    y_pred = rede.predict(X_test_norm)
    probs = rede.predict_proba(X_test_norm)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(10):
        axes[i].imshow(X_test[i].reshape(8, 8), cmap='gray')
        axes[i].set_title(f'Real: {y_test[i]}, Pred: {y_pred[i]}\n'
                         f'Conf: {probs[i, y_pred[i]]:.2%}',
                         fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predicoes_digits.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico salvo em: predicoes_digits.png")
    plt.show()
    
    # plotar convergência
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, rede.n_iteracoes, 100), rede.historico_custo, 
             linewidth=2, marker='o', color='red')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Custo (Cross-Entropy)', fontsize=12)
    plt.title('Convergência - Digits Dataset', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('convergencia_digits.png', dpi=100, bbox_inches='tight')
    print(f"   - Gráfico salvo em: convergencia_digits.png")
    plt.show()
    
    return rede, X_test_norm, y_test, digits

# comparando arquiteturas de rede
def comparar_arquiteturas():
    print("\n\n" + "=" * 60)
    print("TESTE 3: COMPARANDO ARQUITETURAS")
    print("=" * 60)
    
    # usar Iris para comparação
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    X_train_norm, X_test_norm = normalizar_dados(X_train, X_test)
    
    # diferentes arquiteturas
    arquiteturas = [
        [4, 3],           # Sem camada oculta
        [4, 8, 3],        # 1 camada oculta
        [4, 16, 8, 3],    # 2 camadas ocultas
        [4, 32, 16, 8, 3] # 3 camadas ocultas
    ]
    
    print("\n1. Testando arquiteturas...")
    resultados = []
    
    for arq in arquiteturas:
        print(f"\n   Arquitetura: {arq}")
        rede = RedeNeural(camadas=arq, learning_rate=0.1, 
                         n_iteracoes=2000, ativacao='sigmoid')
        rede.fit(X_train_norm, y_train)
        
        acc_train = rede.score(X_train_norm, y_train)
        acc_test = rede.score(X_test_norm, y_test)
        
        print(f"   - Treino: {acc_train:.2%}, Teste: {acc_test:.2%}")
        
        resultados.append({
            'arquitetura': str(arq),
            'acc_train': acc_train,
            'acc_test': acc_test,
            'historico': rede.historico_custo
        })
    
    # plotar comparação
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # acurácias
    x_pos = np.arange(len(arquiteturas))
    accs_train = [r['acc_train'] for r in resultados]
    accs_test = [r['acc_test'] for r in resultados]
    
    width = 0.35
    ax1.bar(x_pos - width/2, accs_train, width, label='Treino', alpha=0.8)
    ax1.bar(x_pos + width/2, accs_test, width, label='Teste', alpha=0.8)
    ax1.set_xlabel('Arquitetura', fontsize=12)
    ax1.set_ylabel('Acurácia', fontsize=12)
    ax1.set_title('Comparação de Acurácias', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([r['arquitetura'] for r in resultados], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # convergência
    for r in resultados:
        ax2.plot(range(0, 2000, 100), r['historico'], 
                linewidth=2, marker='o', label=r['arquitetura'])
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Custo', fontsize=12)
    ax2.set_title('Comparação de Convergência', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacao_arquiteturas.png', dpi=100, bbox_inches='tight')
    print(f"\n2. Gráfico salvo em: comparacao_arquiteturas.png")
    plt.show()

# comparando sigmoid e ReLu
def comparar_ativacoes():
    print("\n\n" + "=" * 60)
    print("TESTE 4: COMPARANDO FUNÇÕES DE ATIVAÇÃO")
    print("=" * 60)
    
    # usar Iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    X_train_norm, X_test_norm = normalizar_dados(X_train, X_test)
    
    print("\n1. Treinando com Sigmoid...")
    rede_sigmoid = RedeNeural(camadas=[4, 16, 8, 3], learning_rate=0.1, 
                             n_iteracoes=2000, ativacao='sigmoid')
    rede_sigmoid.fit(X_train_norm, y_train)
    acc_sigmoid = rede_sigmoid.score(X_test_norm, y_test)
    print(f"   - Acurácia: {acc_sigmoid:.2%}")
    
    print("\n2. Treinando com ReLU...")
    rede_relu = RedeNeural(camadas=[4, 16, 8, 3], learning_rate=0.01, 
                          n_iteracoes=2000, ativacao='relu')
    rede_relu.fit(X_train_norm, y_train)
    acc_relu = rede_relu.score(X_test_norm, y_test)
    print(f"   - Acurácia: {acc_relu:.2%}")
    
    # plotar comparação
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, 2000, 100), rede_sigmoid.historico_custo, 
            linewidth=2, marker='o', label=f'Sigmoid (Acc: {acc_sigmoid:.2%})')
    plt.plot(range(0, 2000, 100), rede_relu.historico_custo, 
            linewidth=2, marker='s', label=f'ReLU (Acc: {acc_relu:.2%})')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Custo', fontsize=12)
    plt.title('Sigmoid vs ReLU', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sigmoid_vs_relu.png', dpi=100, bbox_inches='tight')
    print(f"\n3. Gráfico salvo em: sigmoid_vs_relu.png")
    plt.show()


def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "REDES NEURAIS - TESTES COMPLETOS")
    print("=" * 70)
    
    # executar todos os testes
    teste_iris()
    teste_digits()
    comparar_arquiteturas()
    comparar_ativacoes()
    
    print("\n\n" + "=" * 70)
    print("TODOS OS TESTES CONCLUÍDOS!")
    print("=" * 70)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
