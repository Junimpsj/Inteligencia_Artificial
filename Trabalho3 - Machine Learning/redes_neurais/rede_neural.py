import numpy as np

# iniciando a Rede Neural
# camadas: lista com número de neurônios em cada camada
# ex: [4, 8, 3] = 4 inputs, 1 camada oculta com 8 neurônios, 3 outputs
# learning_rate: taxa de aprendizado
# n_iteracoes: número de épocas de treino
# ativacao: função de ativação simgmoid ou relu
class RedeNeural:
    def __init__(self, camadas, learning_rate=0.01, n_iteracoes=1000, ativacao='sigmoid'):
        self.camadas = camadas
        self.learning_rate = learning_rate
        self.n_iteracoes = n_iteracoes
        self.ativacao = ativacao
        self.pesos = []
        self.bias = []
        self.historico_custo = []
        
        # inicializa pesos e bias aleatoriamente
        self._inicializar_parametros()
    
    # inicializando pesos e bias da rede
    # He para ReLU e Xavier para sigmoid
    def _inicializar_parametros(self):
        np.random.seed(42)
        
        for i in range(len(self.camadas) - 1):
            n_entrada = self.camadas[i]
            n_saida = self.camadas[i + 1]
            
            # inicialização adequada baseada na ativação
            if self.ativacao == 'relu' and i < len(self.camadas) - 2:
                # he initialization para ReLU
                peso = np.random.randn(n_entrada, n_saida) * np.sqrt(2.0 / n_entrada)
            else:
                # xavier initialization para sigmoid
                peso = np.random.randn(n_entrada, n_saida) * np.sqrt(1.0 / n_entrada)
            
            self.pesos.append(peso)
            self.bias.append(np.zeros((1, n_saida)))
    
    # função sigmoid
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    # derivada da sigmoid
    def _derivada_sigmoid(self, a):
        return a * (1 - a)
    
    # função ReLU
    def _relu(self, z):
        return np.maximum(0, z)
    
    # derivada da ReLU
    def _derivada_relu(self, z):
        return (z > 0).astype(float)
    
    # função softmax para camada de saída
    def _softmax(self, z):
        # subtrai o máximo para estabilidade numérica
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # propagação forward (feedforward)
    # retorna todas as ativações para usar no backpropagation
    def _forward_propagation(self, X):
        ativacoes = [X]
        zs = []
        
        for i in range(len(self.pesos)):
            # calcula z = W*a + b
            z = np.dot(ativacoes[-1], self.pesos[i]) + self.bias[i]
            zs.append(z)
            
            # aplica função de ativação
            if i == len(self.pesos) - 1:
                # última camada: softmax para classificação multiclasse
                a = self._softmax(z)
            else:
                # camadas ocultas: usa função escolhida
                if self.ativacao == 'relu':
                    a = self._relu(z)
                else:
                    a = self._sigmoid(z)
            
            ativacoes.append(a)
        
        return ativacoes, zs
    
    # propagação backward (backpropagation)
    # calcula gradientes para atualizar pesos e bias
    def _backward_propagation(self, X, y, ativacoes, zs):
        m = X.shape[0]
        n_camadas = len(self.pesos)
        
        # converte y para one-hot encoding se necessário
        if len(y.shape) == 1:
            n_classes = self.camadas[-1]
            y_one_hot = np.zeros((m, n_classes))
            y_one_hot[np.arange(m), y] = 1
        else:
            y_one_hot = y
        
        # listas para armazenar gradientes
        gradientes_pesos = [None] * n_camadas
        gradientes_bias = [None] * n_camadas
        
        # erro da última camada (com softmax e cross-entropy)
        delta = ativacoes[-1] - y_one_hot
        
        # backpropagation
        for i in reversed(range(n_camadas)):
            # gradientes dos pesos e bias
            gradientes_pesos[i] = (1/m) * np.dot(ativacoes[i].T, delta)
            gradientes_bias[i] = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            # propaga o erro para camada anterior
            if i > 0:
                delta = np.dot(delta, self.pesos[i].T)
                
                # multiplica pela derivada da função de ativação
                if self.ativacao == 'relu':
                    delta *= self._derivada_relu(zs[i-1])
                else:
                    delta *= self._derivada_sigmoid(ativacoes[i])
        
        return gradientes_pesos, gradientes_bias
    
    # calculando o custo (cross-entropy loss)
    def _calcular_custo(self, y_real, y_pred):
        m = y_real.shape[0]
        
        # converte y para one-hot se necessário
        if len(y_real.shape) == 1:
            n_classes = self.camadas[-1]
            y_one_hot = np.zeros((m, n_classes))
            y_one_hot[np.arange(m), y_real] = 1
        else:
            y_one_hot = y_real
        
        # previne log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        custo = -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=1))
        return custo
    
    # treinando a rede neural
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # treinamento
        for epoca in range(self.n_iteracoes):
            # forward propagation
            ativacoes, zs = self._forward_propagation(X)
            
            # backward propagation
            gradientes_pesos, gradientes_bias = self._backward_propagation(X, y, ativacoes, zs)
            
            # atualiza pesos e bias
            for i in range(len(self.pesos)):
                self.pesos[i] -= self.learning_rate * gradientes_pesos[i]
                self.bias[i] -= self.learning_rate * gradientes_bias[i]
            
            # salva custo para visualização
            if epoca % 100 == 0:
                custo = self._calcular_custo(y, ativacoes[-1])
                self.historico_custo.append(custo)
    
    # retornando probabilidades para cada classe
    def predict_proba(self, X):
        X = np.array(X)
        ativacoes, _ = self._forward_propagation(X)
        return ativacoes[-1]
    
    # predizindo classes
    def predict(self, X):
        probabilidades = self.predict_proba(X)
        return np.argmax(probabilidades, axis=1)
    
    # calculando a acurácia do modelo
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
