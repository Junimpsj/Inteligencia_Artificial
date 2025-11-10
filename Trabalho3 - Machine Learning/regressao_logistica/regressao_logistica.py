import numpy as np

# iniciando a Regressão Logística
# learning_rate: taxa de aprendizado para gradiente descendente
# n_iteracoes: número de iterações do treinamento
class RegressaoLogistica:
    def __init__(self, learning_rate=0.01, n_iteracoes=1000):
        self.learning_rate = learning_rate
        self.n_iteracoes = n_iteracoes
        self.pesos = None
        self.bias = None
        self.historico_custo = []
    
    # função sigmoid (logística)
    def _sigmoid(self, z):
        # previne overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    # calculando a função de custo (log loss / cross-entropy)
    def _calcular_custo(self, y_real, y_pred):
        # previne log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        custo = -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))
        return custo
    
    # treinando o modelo usando gradiente descendente
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_amostras, n_features = X.shape
        
        # inicializa pesos e bias
        self.pesos = np.zeros(n_features)
        self.bias = 0
        
        # gradiente descendente
        for i in range(self.n_iteracoes):
            # predição linear
            z = np.dot(X, self.pesos) + self.bias
            # aplica sigmoid
            y_pred = self._sigmoid(z)
            
            # calcula gradientes
            dw = (1 / n_amostras) * np.dot(X.T, (y_pred - y))
            db = (1 / n_amostras) * np.sum(y_pred - y)
            
            # atualiza parâmetros
            self.pesos -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # salva custo para visualização
            if i % 100 == 0:
                custo = self._calcular_custo(y, y_pred)
                self.historico_custo.append(custo)
    
    # retornando probabilidades de pertencer à classe 1
    def predict_proba(self, X):
        X = np.array(X)
        z = np.dot(X, self.pesos) + self.bias
        return self._sigmoid(z)
    
    # predizendo classes (0 ou 1)
    def predict(self, X, threshold=0.5):
        probabilidades = self.predict_proba(X)
        return (probabilidades >= threshold).astype(int)
    
    # calculando a acurácia do modelo
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
