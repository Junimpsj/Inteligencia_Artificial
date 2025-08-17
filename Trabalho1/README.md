# **Trabalho 1 – Algoritmos de Busca sem Informação**

## 📌 Descrição Geral
Este projeto implementa diferentes **algoritmos de busca sem informação**, aplicados a **problemas práticos simulados**, usando **linguagem C pura** (sem bibliotecas externas).


Cada algoritmo foi contextualizado em um cenário distinto, buscando não só implementar corretamente, mas também mostrar **criatividade e aplicabilidade real**.

---

## 📂 Estrutura do Projeto

| Arquivo | Algoritmo | Cenário |
|--------|-----------|---------|
| `BuscaEmLargura_BFS.c` | **BFS** | Resgate em prédio com múltiplos andares |
| `BuscaComCustoUniforme_UCS1.c` | **UCS** | Rota ótima para drone entregador |
| `BuscaComCustoUniforme_UCS2.c` | **UCS** | Montagem de PC com melhor custo-benefício |
| `BuscaEmProfundidade_DFS.c` | **DFS** | Análise de árvore genealógica familiar |
| `BuscaEmProfundidadeLimitada_DLS.c` | **DLS** | Diagnóstico médico baseado em sintomas |
| `BuscaEmProfundidadeIterativa_IDS.c` | **IDS** | Planejamento de viagem com atrações obrigatórias |

---

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** C (padrão ANSI)
- **Bibliotecas:** `stdio.h`, `stdlib.h`, `string.h`, `time.h`
- **Paradigma:** Estruturado
- **Execução:** Terminal (CLI)

---

## 📜 Resumo dos Algoritmos

### ✅ BFS – Resgate em Prédio
- Explora o menor caminho até a vítima.
- Evita o maior número possível de portas.
- Ideal para ambiente com obstáculos previsíveis.

### ✅ UCS – Drone Entregador
- Rota ótima entre locais, minimizando energia gasta.
- Considera pesos (custos) diferentes nas arestas.

### ✅ UCS – Montagem de PC
- Busca configuração de PC com melhor desempenho/custo.
- Considera CPU, GPU, RAM, etc., com orçamento limitado.

### ✅ DFS – Árvore Genealógica
- Navega pela árvore de ancestrais.
- Encontra o mais distante e o caminho até ele.

### ✅ DLS – Diagnóstico Médico
- Busca causas e doenças prováveis com limite de profundidade.
- Útil quando o sistema não pode analisar todos os caminhos.

### ✅ IDS – Planejamento de Viagem
- Planeja rota entre cidades passando por atrações obrigatórias.
- Aumenta a profundidade progressivamente até encontrar solução.

---

## 🚀 Como Compilar e Executar

Cada código é independente. Compile com:

```bash
gcc NomeDoArquivo.c -o exec
./exec
```

## Comparativo dos algoritmos

| Algoritmo | Completo? | Ótimo? | Memória | Tempo  |
| --------- | --------- | ------ | ------- | ------ |
| BFS       | ✅         | ✅      | Alto    | Médio  |
| UCS       | ✅         | ✅      | Médio   | Médio  |
| DFS       | ✅         | ❌      | Baixo   | Rápido |
| DLS       | ❌         | ❌      | Baixo   | Rápido |
| IDS       | ✅         | ✅      | Baixo   | Lento  |