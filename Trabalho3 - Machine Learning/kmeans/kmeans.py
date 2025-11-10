import numpy as np

# iniciando o K-Means
# k: número de clusters
# max_iteracoes: número máximo de iterações
# seed: semente para reprodutibilidade
class KMeans:
    def __init__(self, k=3, max_iteracoes=100, seed=42):
        self.k = k
        self.max_iteracoes = max_iteracoes
        self.seed = seed
        self.centroides = None
        self.labels = None
        self.inertia = None
    
    # treinando o modelo K-Means
    def fit(self, X):
        np.random.seed(self.seed)
        X = np.array(X)
        n_amostras = X.shape[0]
        
        # inicializa centroides aleatoriamente
        indices_aleatorios = np.random.choice(n_amostras, self.k, replace=False)
        self.centroides = X[indices_aleatorios]
        
        # iterações do K-Means
        for iteracao in range(self.max_iteracoes):
            # atribui cada ponto ao centroide mais próximo
            self.labels = self._atribuir_clusters(X)
            
            # salva centroides antigos para verificar convergência
            centroides_antigos = self.centroides.copy()
            
            # atualiza centroides (média dos pontos de cada cluster)
            self.centroides = self._atualizar_centroides(X, self.labels)
            
            # verifica convergência (centroides não mudam)
            if np.allclose(centroides_antigos, self.centroides):
                break
        
        # calcula inércia final (métrica de qualidade)
        self.inertia = self._calcular_inertia(X, self.labels)
    
    # atribuindo cada ponto ao cluster do centroide mais próximo
    def _atribuir_clusters(self, X):
        distancias = np.zeros((X.shape[0], self.k))
        
        # calcula distância de cada ponto para cada centroide
        for i, centroide in enumerate(self.centroides):
            distancias[:, i] = np.sqrt(np.sum((X - centroide) ** 2, axis=1))
        
        # retorna índice do centroide mais próximo
        return np.argmin(distancias, axis=1)
    
    # atualizando centroides como a média dos pontos de cada cluster
    def _atualizar_centroides(self, X, labels):
        novos_centroides = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            pontos_cluster = X[labels == i]
            if len(pontos_cluster) > 0:
                novos_centroides[i] = pontos_cluster.mean(axis=0)
            else:
                # se cluster vazio, mantém centroide atual
                novos_centroides[i] = self.centroides[i]
        
        return novos_centroides
    
    # calculando a inércia (soma das distâncias ao quadrado aos centroides)
    def _calcular_inertia(self, X, labels):
        inertia = 0
        for i in range(self.k):
            pontos_cluster = X[labels == i]
            if len(pontos_cluster) > 0:
                inertia += np.sum((pontos_cluster - self.centroides[i]) ** 2)
        return inertia
    
    # predizindo o cluster para novos dados
    def predict(self, X):
        X = np.array(X)
        return self._atribuir_clusters(X)
    
    # método do cotovelo para encontrar o K ideal
    # retorna lista de inércias para cada k
    def metodo_cotovelo(self, X, k_max=10):
        inercias = []
        
        for k in range(1, k_max + 1):
            kmeans_temp = KMeans(k=k, max_iteracoes=self.max_iteracoes, seed=self.seed)
            kmeans_temp.fit(X)
            inercias.append(kmeans_temp.inertia)
        
        return inercias
