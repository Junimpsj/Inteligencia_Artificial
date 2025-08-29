#executa um grid de hiperparâmetros para blackjack com q-learning
#avalia a política treinada e salva um CSV com os resultados.

#como usar:
#   python experiments.py --alphas 0.05,0.1,0.2 --episodes 50000,100000,200000 --gammas 1.0 --repeats 2
#   python experiments.py --save-curves --out results.csv

from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from typing import List

from qlearning import train_q_learning, evaluate_policy
from analysis_utils import save_learning_curve  # opcional, funciona mesmo sem matplotlib

def _parse_float_list(s: str) -> List[float]:
    #transforma string tipo "0.1,0.2" em lista de floats
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_int_list(s: str) -> List[int]:
    #transforma string tipo "100,200" em lista de ints
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="Grid de experimentos para Blackjack Q-learning.")
    parser.add_argument("--alphas", type=str, default="0.05,0.1,0.2", help="lista de alphas, sep por vírgula")
    parser.add_argument("--episodes", type=str, default="50000,100000,200000", help="lista de episódios, sep por vírgula")
    parser.add_argument("--gammas", type=str, default="1.0", help="lista de gammas, sep por vírgula")
    parser.add_argument("--eps-start", type=float, default=1.0, dest="eps_start", help="ε inicial")
    parser.add_argument("--eps-end", type=float, default=0.05, dest="eps_end", help="ε mínimo")
    parser.add_argument("--eps-decay", type=float, default=0.9995, dest="eps_decay", help="decaimento multiplicativo de ε")
    parser.add_argument("--repeats", type=int, default=1, help="repetições por configuração (seeds diferentes)")
    parser.add_argument("--base-seed", type=int, default=42, help="seed base; cada repetição soma +rep_idx")
    parser.add_argument("--eval-episodes", type=int, default=100_000, help="nº episódios para avaliação greedy")
    parser.add_argument("--out", type=str, default="experiments_results.csv", help="arquivo CSV de saída")
    parser.add_argument("--append", action="store_true", help="acrescenta ao CSV se já existir (senão sobrescreve)")
    parser.add_argument("--save-curves", action="store_true", help="salva curvas de aprendizagem por execução")
    parser.add_argument("--curves-dir", type=str, default="curves", help="pasta para curvas (se --save-curves)")

    args = parser.parse_args()

    alphas = _parse_float_list(args.alphas)
    episodes_list = _parse_int_list(args.episodes)
    gammas = _parse_float_list(args.gammas)

    if args.save_curves and not os.path.isdir(args.curves_dir):
        os.makedirs(args.curves_dir, exist_ok=True)

    header = [
        "alpha", "gamma", "episodes", "eps_start", "eps_end", "eps_decay", "seed",
        "win_rate", "draw_rate", "loss_rate", "avg_return",
        "wins", "losses", "draws",
        "train_time_s", "eval_time_s", "curve_path"
    ]

    write_header = True
    if args.append and os.path.exists(args.out):
        write_header = False

    with open(args.out, "a" if args.append else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        #produto cartesiano das combinações
        for (alpha, gamma, n_episodes) in itertools.product(alphas, gammas, episodes_list):
            for rep in range(args.repeats):
                seed = args.base_seed + rep

                print(f"== Rodando: alpha={alpha}, gamma={gamma}, episodes={n_episodes}, seed={seed} ==")

                t0 = time.perf_counter()
                Q, stats = train_q_learning(
                    num_episodes=n_episodes,
                    alpha=alpha,
                    gamma=gamma,
                    eps_start=args.eps_start,
                    eps_end=args.eps_end,
                    eps_decay=args.eps_decay,
                    seed=seed,
                )
                t1 = time.perf_counter()

                ev = evaluate_policy(Q, n_episodes=args.eval_episodes, seed=seed + 10_000)
                t2 = time.perf_counter()

                train_time = t1 - t0
                eval_time = t2 - t1

                curve_path = ""
                if args.save_curves:
                    curve_name = f"curve_alpha{alpha}_gamma{gamma}_eps{n_episodes}_seed{seed}.png"
                    curve_path = os.path.join(args.curves_dir, curve_name)
                    try:
                        save_learning_curve(stats["episode_rewards"], path=curve_path)
                    except Exception as e:
                        print(f"[Aviso] Falha ao salvar curva ({e}). Prosseguindo sem curva.")
                        curve_path = ""

                row = [
                    alpha, gamma, n_episodes, args.eps_start, args.eps_end, args.eps_decay, seed,
                    ev["win_rate"], ev["draw_rate"], ev["loss_rate"], ev["avg_return"],
                    stats["wins"], stats["losses"], stats["draws"],
                    round(train_time, 4), round(eval_time, 4), curve_path
                ]
                writer.writerow(row)
                #também imprime um resumo no terminal
                print(f"  -> win={ev['win_rate']:.4f} draw={ev['draw_rate']:.4f} "
                      f"loss={ev['loss_rate']:.4f} avg_return={ev['avg_return']:.4f} "
                      f"| wins={stats['wins']} losses={stats['losses']} draws={stats['draws']} "
                      f"| train_time={train_time:.2f}s eval_time={eval_time:.2f}s")
                if curve_path:
                    print(f"  -> curva salva em: {curve_path}")

    print(f"\n[OK] Resultados salvos em: {args.out}")
    #aqui acabou, vai analisar o CSV agora

if __name__ == '__main__':
    main()