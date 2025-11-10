import numpy as np

# iniciando o regressor KNN
# k: número de vizinhos a considerar
class KNNRegressor:
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
    
    # encontrando os K vizinhos mais próximos e retorna seus valores
    def _encontrar_vizinhos(self, x):
        distancias = []
        for i, x_train in enumerate(self.X_train):
            dist = self._calcular_distancia_euclidiana(x, x_train)
            distancias.append((dist, self.y_train[i]))
        
        # ordena por distância e pega os K primeiros
        distancias.sort(key=lambda x: x[0])
        valores_vizinhos = [dist[1] for dist in distancias[:self.k]]
        
        return valores_vizinhos
    
    # predizindo valores para novos dados (média dos vizinhos)
    def predict(self, X):
        X = np.array(X)
        predicoes = []
        
        for x in X:
            # encontra vizinhos e calcula a média
            valores_vizinhos = self._encontrar_vizinhos(x)
            predicao = np.mean(valores_vizinhos)
            predicoes.append(predicao)
        
        return np.array(predicoes)
    
    # calculando o R² (coeficiente de determinação)
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    # calculando o erro quadrático médio (MSE)
    def mean_squared_error(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse
