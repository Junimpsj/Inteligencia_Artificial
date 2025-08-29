#ambiente de Blackjack simplificado

from typing import List, Tuple
import random

__all__ = ["draw_card", "hand_value", "is_bust", "BlackjackEnv"]

#utilidades de cartas ----------------------------
def draw_card() -> int:
    #retorna valor de uma carta: 1..10 (J/Q/K = 10).
    #rão tem coringa
    c = random.randint(1, 13)
    return min(c, 10)

def hand_value(cards: List[int]) -> Tuple[int, bool]:
    """
    Retorna (total, usable_ace).
    usable_ace = True quando contar um Ás como 11 mantém total <= 21.
    Se tiver Ás, pode ser 1 ou 11, depende do humor
    """
    total = sum(cards)
    usable_ace = (1 in cards) and (total + 10 <= 21)
    if usable_ace:
        total += 10
    return total, usable_ace

def is_bust(total: int) -> bool:
    #passou de 21
    return total > 21

#ambiente -----------------------
class BlackjackEnv:
    """
    Estado: (player_sum, dealer_upcard, usable_ace[0/1])
    Ações: 0=parar (stick), 1=pedir (hit)
    Recompensa: +1 vitória, 0 empate, -1 derrota (episódio termina)
    """
    def __init__(self):
        self.player: List[int] = []
        self.dealer: List[int] = []
        self.done: bool = False

    def reset(self):
        #começa com duas cartas pra cada
        self.player = [draw_card(), draw_card()]
        self.dealer = [draw_card(), draw_card()]
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        #observa o estado atual
        p_sum, p_ace = hand_value(self.player)
        d_up = self.dealer[0]
        return (p_sum, d_up, int(p_ace))

    def step(self, action: int):
        assert not self.done, "Episódio terminado. Chame reset()."

        #ação do do jogador
        if action == 1:  # hit
            self.player.append(draw_card())
            p_sum, _ = hand_value(self.player)
            if is_bust(p_sum):
                self.done = True
                # rint("Jogador estourou!")
                return self._get_obs(), -1.0, True  #estouro do jogador
            return self._get_obs(), 0.0, False

        #stick -> dealer compra até 17+
        self.done = True
        d_sum, _ = hand_value(self.dealer)
        while d_sum < 17:
            self.dealer.append(draw_card())
            d_sum, _ = hand_value(self.dealer)
            if is_bust(d_sum):
                #print("Dealer estourou!")
                return self._get_obs(), +1.0, True  #dealer estourou

        p_sum, _ = hand_value(self.player)
        if p_sum > d_sum:
            return self._get_obs(), +1.0, True
        if p_sum < d_sum:
            return self._get_obs(), -1.0, True
        return self._get_obs(), 0.0, True