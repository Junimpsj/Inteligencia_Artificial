#ponto de entrada: treino, avaliação e geração de saídas

import argparse
from qlearning import train_q_learning, evaluate_policy
from analysis_utils import save_learning_curve, learned_policy_table, print_policy_ascii

def main():
    parser = argparse.ArgumentParser(description="Q-learning em Blackjack (modular).")
    parser.add_argument("--episodes", type=int, default=200000, help="nº de episódios de treino")
    parser.add_argument("--alpha", type=float, default=0.1, help="taxa de aprendizado α")
    parser.add_argument("--gamma", type=float, default=1.0, help="fator de desconto γ")
    parser.add_argument("--eps_start", type=float, default=1.0, help="ε inicial")
    parser.add_argument("--eps_end", type=float, default=0.05, help="ε mínimo")
    parser.add_argument("--eps_decay", type=float, default=0.9995, help="decaimento de ε por episódio")
    args = parser.parse_args()

    print("Treinando...")
    Q, stats = train_q_learning(
        num_episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        seed=42,
    )
    print(f"Treino concluído com {args.episodes} episódios.")
    print(f"Wins: {stats['wins']} | Losses: {stats['losses']} | Draws: {stats['draws']}")

    print("Avaliando política (greedy)...")
    ev = evaluate_policy(Q, n_episodes=100_000, seed=7)
    for k, v in ev.items():
        print(f"{k}: {v:.4f}")

    save_learning_curve(stats["episode_rewards"], "learning_curve.png")

    no_ace = learned_policy_table(Q, usable_ace=False)
    yes_ace = learned_policy_table(Q, usable_ace=True)
    print_policy_ascii(no_ace, "Política aprendida — SEM Ás utilizável (S=parar, H=pedir)")
    print_policy_ascii(yes_ace, "Política aprendida — COM Ás utilizável (S=parar, H=pedir)")

if __name__ == "__main__":
    main()