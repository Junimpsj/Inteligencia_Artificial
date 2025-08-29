#funções de análise: curva de aprendizagem e política aprendida
#aqui é só pra mostrar gráfico e tabela

from typing import Dict, Tuple
import numpy as np

#matplotlib é opcional (somente para o gráfico)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

State = Tuple[int, int, int]

def moving_average(x: np.ndarray, window: int = 5000) -> np.ndarray:
    #média móvel
    if len(x) < window:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[window:] - c[:-window]) / float(window)

def save_learning_curve(rewards: np.ndarray, path: str = "learning_curve.png"):
    #salva o gráfico, se der
    if not HAS_MPL:
        print("[Aviso] matplotlib não encontrado; gráfico não será salvo.")
        return
    w = max(10, min(5000, len(rewards)//20))
    ma = moving_average(rewards, window=w)
    plt.figure()
    plt.plot(ma)
    plt.title(f"Recompensa média móvel (janela={w})")
    plt.xlabel("blocos")
    plt.ylabel("retorno médio")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[OK] Gráfico salvo em: {path}")

def learned_policy_table(Q: Dict[State, np.ndarray], usable_ace: bool) -> np.ndarray:
    #monta a tabela da política aprendida, tipo aquelas do livro
    table = np.zeros((10, 10), dtype=int)  # linhas: player 12..21; colunas: dealer 1..10
    for p_sum in range(12, 22):
        for d_up in range(1, 11):
            s = (p_sum, d_up, int(usable_ace))
            a = int(np.argmax(Q[s]))
            table[p_sum - 12, d_up - 1] = a  #0=parar (S), 1=pedir (H)
    return table

def print_policy_ascii(table: np.ndarray, title: str):
    #imprime a política no terminal
    print("\n" + title)
    header = "P\\D | " + " ".join(f"{d:>2}" for d in range(1, 11))
    print(header)
    print("-" * len(header))
    for i, p_sum in enumerate(range(12, 22)):
        row = " ".join(" S" if a == 0 else " H" for a in table[i])
        print(f"{p_sum:>3} | {row}")
    #print("Fim da tabela!")