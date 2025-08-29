# Blackjack RL (Q-learning)

Implementação didática do jogo **Blackjack** e de um agente **Q-learning** que aprende a jogar. O projeto está dividido em módulos para facilitar o entendimento, manutenção e experimentação.

## Objetivos
1. Implementar um ambiente simplificado de Blackjack.
2. Treinar um agente Q-learning para jogar.
3. Comparar resultados variando hiperparâmetros e gerar gráficos/tabelas para relatório.

---

## Requisitos
- Python 3.9+
- `numpy`
- `matplotlib` (opcional, só para gráficos)

Instalação rápida:
```bash
pip install numpy matplotlib
```

---

## Estrutura do projeto

```
Trabalho2 - Q-learning/
├─ env_blackjack.py       # ambiente e utilidades de cartas
├─ qlearning.py           # Q-learning (treino) + avaliação
├─ analysis_utils.py      # análise: curva de aprendizagem e política aprendida
├─ main.py                # ponto de entrada para treinar/avaliar
├─ experiments.py         # grid search de hiperparâmetros e geração de CSV
```

### O que cada módulo faz
- **env_blackjack.py**  
  - Funções: `draw_card()`, `hand_value()`, `is_bust()`
  - Classe: `BlackjackEnv` com métodos `reset()` e `step(action)`
  - Estado: `(soma_jogador, carta_aberta_dealer, ace_utilizavel[0/1])`
  - Ações: `0=parar (stick)`, `1=pedir (hit)`
  - Dealer compra até total ≥ 17. Baralho infinito.
- **qlearning.py**  
  - `epsilon_greedy(Q, state, epsilon)`: política de exploração
  - `train_q_learning(...)`: treina Q-table com TD(0)
  - `evaluate_policy(Q, ...)`: avalia política greedy
- **analysis_utils.py**  
  - `moving_average(...)`, `save_learning_curve(...)`: gráfico da recompensa média móvel
  - `learned_policy_table(Q, usable_ace)`: matriz de ações ótimas
  - `print_policy_ascii(table, title)`: imprime política em ASCII
- **main.py**  
  - Faz o parse dos argumentos, treina, avalia, salva gráfico e imprime tabelas
- **experiments.py**  
  - Executa grid de hiperparâmetros, avalia políticas e salva resultados em CSV

---

## Como rodar

### Treino padrão + avaliação
No diretório do projeto:
```bash
python main.py --episodes 200000 --alpha 0.1 --gamma 1.0
```
Saídas:
- Métricas no terminal (taxa de vitória/empate/derrota e retorno médio)
- `learning_curve.png`: curva da recompensa média móvel
- Tabelas da política aprendida (com e sem Ás utilizável), em ASCII

### Ajustando hiperparâmetros
Todos têm flags na CLI:
```bash
python main.py --episodes 100000 --alpha 0.05 --gamma 1.0 --eps_start 1.0 --eps_end 0.05 --eps_decay 0.9995
```

### Rodando experimentos e gerando CSV
```bash
python experiments.py --alphas 0.05,0.1,0.2 --episodes 50000,100000,200000 --gammas 1.0 --repeats 2 --save-curves --out resultados.csv
```
O CSV terá métricas para cada configuração testada.

---

## Metodologia (resumo)

- Estado: `(player_sum, dealer_upcard, usable_ace)`
- Ações: `stick`/`hit` (`0/1`)
- Transições: jogador age; dealer compra até ≥ 17; episódio termina se alguém estourar
- Baralho infinito; J/Q/K=10; Ás=1 ou 11 (usável se total ≤ 21)
- Recompensa terminal: `+1 / 0 / −1`
- Aprendizado: Q-learning tabular, política ε-gulosa com decaimento
- Avaliação: política greedy em amostragem independente

---

## Dicas rápidas
- Os módulos fixam `seed` para reprodutibilidade.
- 50k episódios já mostra tendência; 200k estabiliza melhor.
- Espere `win_rate < 0.5` (Blackjack favorece o dealer).
- Resultados variando muito? Aumente episódios ou reduza `α`.

---

## Problemas comuns
- `ModuleNotFoundError`: rode os comandos a partir da pasta do projeto.
- Sem gráfico? Instale `matplotlib` ou ignore (o resto funciona).
- Resultados instáveis: aumente episódios, reduza `α`, mantenha decaimento de `ε`.

---
- **Tempo de treino**: 50k episódios já mostra tendência; 200k estabiliza melhor.  
- **Interpretação**: espere `win_rate < 0.5` (o blackjack padrão é desfavorável ao jogador). O foco é a **melhora** da política e o impacto dos parâmetros.

---

## Solução de problemas
- **`ModuleNotFoundError`**: rode os comandos a partir da pasta `blackjack_rl/` ou ajuste `PYTHONPATH`.  
- **Sem gráfico**: instale `matplotlib` ou ignore (o restante funciona).  
- **Resultados variando muito**: aumente episódios e/ou reduza `α`; mantenha `eps_decay` para explorar no início e estabilizar no fim.