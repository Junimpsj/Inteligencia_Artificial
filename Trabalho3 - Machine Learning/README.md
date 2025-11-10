# Machine Learning - Implementações

Implementações de algoritmos de Machine Learning em Python puro (sem bibliotecas de ML).

## 📚 Algoritmos Implementados

1. **KNN Classificador** - K-Nearest Neighbors para classificação
2. **KNN Regressor** - K-Nearest Neighbors para regressão
3. **K-Means** - Agrupamento (clustering)
4. **Regressão Logística** - Classificação binária
5. **Decision Tree** - Árvore de Decisão
6. **Redes Neurais** - Multilayer Perceptron com backpropagation

## 🛠️ Bibliotecas Permitidas

- **NumPy** - Para operações matemáticas
- **Pandas** - Para manipulação de dados
- **Matplotlib** - Para visualizações
- **Sklearn** - Utilizada APENAS para carregar os datasets (não usar os algoritmos!)

## 📦 Instalação das Dependências

```bash
pip install numpy pandas matplotlib scikit-learn
```

ou

```bash
pip instal requirements.txt
```

## 🚀 Como Executar

Cada algoritmo tem seu próprio diretório com dois arquivos:
- `<algoritmo>.py` - Implementação da classe
- `teste_<algoritmo>.py` - Exemplo de uso com dataset

Para executar um teste:

```bash
# Exemplo: KNN Classificador
cd "knn_classificador"
python teste_knn_classificador.py

# Exemplo: Redes Neurais
cd "redes neurais"
python teste_rede_neural.py
```

## 📊 Onde Encontrar Datasets Públicos

### 1. Sklearn Datasets (Mais Fácil)
```python
from sklearn import datasets

#Classificação
iris = datasets.load_iris()              #Flores (3 classes)
digits = datasets.load_digits()          #Dígitos 0-9
wine = datasets.load_wine()              #Vinhos
breast_cancer = datasets.load_breast_cancer()  #Câncer (binário)

#Regressão
california = datasets.fetch_california_housing()  #Preços de casas
diabetes = datasets.load_diabetes()      #Progressão de diabetes
```

### 2. UCI Machine Learning Repository
- Site: https://archive.ics.uci.edu/ml/
- Baixe arquivos CSV e carregue com pandas:
```python
import pandas as pd
df = pd.read_csv('dataset.csv')
X = df.drop('target_column', axis=1).values
y = df['target_column'].values
```

### 3. OpenML
- Site: https://www.openml.org/
- Integra com sklearn:
```python
from sklearn.datasets import fetch_openml
data = fetch_openml(name='diabetes', version=1)
```

## 📝 Estrutura dos Arquivos

```
Trabalho3 - Machine Learning/
├── knn_classificador/
│   ├── knn_classificador.py
│   └── teste_knn_classificador.py
├── knn_regressor/
│   ├── knn_regressor.py
│   └── teste_knn_regressor.py
├── kmeans/
│   ├── kmeans.py
│   └── teste_kmeans.py
├── regressao_logistica/
│   ├── regressao_logistica.py
│   └── teste_regressao_logistica.py
├── decision_tree/
│   ├── decision_tree.py
│   └── teste_decision_tree.py
└── redes neurais/
    ├── rede_neural.py
    └── teste_rede_neural.py
```

## 🎯 Características das Implementações

### KNN Classificador
- Distância euclidiana
- Voto majoritário dos K vizinhos
- Teste com dataset Iris

### KNN Regressor
- Distância euclidiana
- Média dos valores dos K vizinhos
- Teste com dataset California Housing

### K-Means
- Inicialização aleatória
- Critério de convergência
- Método do cotovelo para escolher K
- Teste com dataset Iris

### Regressão Logística
- Classificação binária
- Gradiente descendente
- Função sigmoid
- Normalização de dados
- Teste com dataset Breast Cancer

### Decision Tree
- Critério Gini para divisões
- Controle de profundidade
- Prevenção de overfitting
- Teste com dataset Iris

### Redes Neurais
- Múltiplas camadas ocultas
- Funções de ativação: Sigmoid e ReLU
- Softmax na saída
- Backpropagation
- Inicialização Xavier/He
- Testes com Iris e Digits

## 📈 Visualizações Geradas

Os scripts de teste geram gráficos automaticamente:
- Curvas de convergência
- Acurácia vs parâmetros
- Decision boundaries
- Matrizes de confusão
- Distribuições de probabilidades
- Método do cotovelo (K-Means)

## 💡 Dicas de Uso

1. **Normalização**: Sempre normalize dados para Regressão Logística e Redes Neurais
2. **Escolha de K**: Use validação cruzada ou método do cotovelo
3. **Learning Rate**: Comece com 0.01-0.1 e ajuste
4. **Profundidade da Árvore**: Cuidado com overfitting, teste valores entre 3-10
5. **Arquitetura da Rede**: Comece simples e aumente se necessário