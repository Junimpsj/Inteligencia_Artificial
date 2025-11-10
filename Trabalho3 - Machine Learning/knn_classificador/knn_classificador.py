import numpy as np
from collections import Counter

# iniciando o classificador KNN
# k: número de vizinhos a considerar
class KNNClassificador:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    # treinando o modelo (apenas armazena os dados de treino)
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    # calculando a distância euclidiana entre dois pontos
    def _calcular_distancia_euclidiana(self, ponto1, ponto2):
        return np.sqrt(np.sum((ponto1 - ponto2) ** 2))
    
    # encontrando os K vizinhos mais próximos de um ponto
    def _encontrar_vizinhos(self, x):
        distancias = []
        for i, x_train in enumerate(self.X_train):
            dist = self._calcular_distancia_euclidiana(x, x_train)
            distancias.append((dist, self.y_train[i]))
        
        # ordena por distância e pega os K primeiros
        distancias.sort(key=lambda x: x[0])
        vizinhos = [dist[1] for dist in distancias[:self.k]]
        
        return vizinhos
    
    # predizendo a classe para novos dados
    def predict(self, X):
        X = np.array(X)
        predicoes = []
        
        for x in X:
            # encontra vizinhos e pega a classe mais comum
            vizinhos = self._encontrar_vizinhos(x)
            classe_mais_comum = Counter(vizinhos).most_common(1)[0][0]
            predicoes.append(classe_mais_comum)
        
        return np.array(predicoes)
    
    # calculando a acurácia do modelo
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
