import numpy as np

# representação do nó da árvore
class No:
    def __init__(self, feature=None, threshold=None, esquerda=None, 
                 direita=None, valor=None):
        self.feature = feature
        self.threshold = threshold
        self.esquerda = esquerda
        self.direita = direita
        self.valor = valor

# iniciando a árvore com os parâmetros
# max_profundidade: profundidade máxima da árvore
# min_amostras_split: número mínimo de amostras para dividir um nó
class DecisionTree:
    def __init__(self, max_profundidade=10, min_amostras_split=2):
        self.max_profundidade = max_profundidade
        self.min_amostras_split = min_amostras_split
        self.raiz = None
    
    # calculando o índice gini (impureza)
    def _calcular_gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        proporcoes = counts / len(y)
        gini = 1 - np.sum(proporcoes ** 2)
        return gini
    
    # dividindo os dados baseado em uma feature e threshold
    def _dividir_dados(self, X, y, feature, threshold):
        mascara_esquerda = X[:, feature] <= threshold
        mascara_direita = X[:, feature] > threshold
        
        return (X[mascara_esquerda], y[mascara_esquerda],
                X[mascara_direita], y[mascara_direita])
    
    # calculando o ganho de informação da divisão
    def _calcular_ganho_informacao(self, y, y_esquerda, y_direita):
        n = len(y)
        n_esquerda = len(y_esquerda)
        n_direita = len(y_direita)
        
        if n_esquerda == 0 or n_direita == 0:
            return 0
        
        # antes da divisão
        gini_pai = self._calcular_gini(y)
        
        # depois da divisão (média ponderada)
        gini_filhos = (n_esquerda / n) * self._calcular_gini(y_esquerda) + \
                      (n_direita / n) * self._calcular_gini(y_direita)
        
        # qual foi a redução de impurezas
        ganho = gini_pai - gini_filhos
        return ganho
    
    # encontrando a melhor divisão (feature e threshold)
    def _encontrar_melhor_divisao(self, X, y):
        melhor_ganho = -1
        melhor_feature = None
        melhor_threshold = None
        
        n_features = X.shape[1]
        
        # testando todas as features
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                _, y_esquerda, _, y_direita = self._dividir_dados(X, y, feature, threshold)
                
                ganho = self._calcular_ganho_informacao(y, y_esquerda, y_direita)
                
                if ganho > melhor_ganho:
                    melhor_ganho = ganho
                    melhor_feature = feature
                    melhor_threshold = threshold
        
        return melhor_feature, melhor_threshold
    
    # construindo a árvore recursivamente
    def _construir_arvore(self, X, y, profundidade=0):
        n_amostras, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # condições de parada
        if (profundidade >= self.max_profundidade or 
            n_classes == 1 or 
            n_amostras < self.min_amostras_split or
            n_amostras == 0):
            if n_amostras == 0:
                # se não há amostras, retorna classe 0
                return No(valor=0)
            classe_mais_comum = np.bincount(y).argmax()
            return No(valor=classe_mais_comum)
        
        melhor_feature, melhor_threshold = self._encontrar_melhor_divisao(X, y)
        
        if melhor_feature is None:
            classe_mais_comum = np.bincount(y).argmax()
            return No(valor=classe_mais_comum)
        
        X_esq, y_esq, X_dir, y_dir = self._dividir_dados(X, y, melhor_feature, 
                                                          melhor_threshold)
        
        # se a divisão resultar em um lado vazio, cria nó folha
        if len(y_esq) == 0 or len(y_dir) == 0:
            classe_mais_comum = np.bincount(y).argmax()
            return No(valor=classe_mais_comum)
        
        subarvore_esquerda = self._construir_arvore(X_esq, y_esq, profundidade + 1)
        subarvore_direita = self._construir_arvore(X_dir, y_dir, profundidade + 1)
        
        return No(feature=melhor_feature, threshold=melhor_threshold,
                 esquerda=subarvore_esquerda, direita=subarvore_direita)
    
    # treinando a árvore com os parâmetros
    # x: features de treino
    # y: labels de treino
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.raiz = self._construir_arvore(X, y)
    
    # aqui estamos percorrendo a árvore para fazer predição de valores
    def _percorrer_arvore(self, x, no):
        if no.valor is not None:
            return no.valor
        
        if x[no.feature] <= no.threshold:
            return self._percorrer_arvore(x, no.esquerda)
        else:
            return self._percorrer_arvore(x, no.direita)
    
    # fazendo predições para novos dados
    def predict(self, X):
        X = np.array(X)
        return np.array([self._percorrer_arvore(x, self.raiz) for x in X])
    
    # calculando a acurácia do modelo
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
