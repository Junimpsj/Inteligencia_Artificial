# Q-learning tabular e avaliação para o Blackjack

from collections import defaultdict
from typing import Dict, Tuple
import random
import numpy as np

from env_blackjack import BlackjackEnv

State = Tuple[int, int, int]  # (player_sum, dealer_upcard, usable_ace)

def epsilon_greedy(Q: Dict[State, np.ndarray], state: State, epsilon: float) -> int:
    #escolhe ação 0/1 com política ε-gulosa.
    #se der sorte, escolhe aleatório, senão vai no melhor (ou não)
    if random.random() < epsilon:
        return random.choice([0, 1])
    return int(np.argmax(Q[state]))

def train_q_learning(
    num_episodes: int = 200_000,
    alpha: float = 0.1,
    gamma: float = 1.0,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.9995,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)

    env = BlackjackEnv()
    Q: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float32))

    episode_rewards = []
    epsilon = eps_start
    wins = losses = draws = 0

    for ep in range(num_episodes):
        s = env.reset()
        done = False
        G = 0.0

        while not done:
            a = epsilon_greedy(Q, s, epsilon)
            s2, r, done = env.step(a)

            #atualização TD(0)
            target = r + (0.0 if done else gamma * np.max(Q[s2]))
            Q[s][a] += alpha * (target - Q[s][a])

            s = s2
            G += r

        episode_rewards.append(G)
        #contagem de vitórias/derrotas/empates, é bom saber
        if G > 0: wins += 1
        elif G < 0: losses += 1
        else: draws += 1

        #epsilon vai diminuindo, mas nunca chega a zero
        epsilon = max(eps_end, epsilon * eps_decay)

    stats = {
        "wins": wins, "losses": losses, "draws": draws,
        "episode_rewards": np.array(episode_rewards, dtype=np.float32),
        "Q": Q
    }
    return Q, stats

def evaluate_policy(Q: Dict[State, np.ndarray], n_episodes: int = 50_000, seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)

    env = BlackjackEnv()
    rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        G = 0.0
        while not done:
            a = int(np.argmax(Q[s]))
            s, r, done = env.step(a)
            G += r
        rewards.append(G)
    rewards = np.array(rewards, dtype=np.float32)
    return {
        "avg_return": float(rewards.mean()),
        "win_rate": float((rewards > 0).mean()),
        "draw_rate": float((rewards == 0).mean()),
        "loss_rate": float((rewards < 0).mean()),
    }