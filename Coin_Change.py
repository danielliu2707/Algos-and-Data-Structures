# Implementation of Coins Change using bottom-up approach and reconstructing solution using backtracking:

def coin_change(coins, V):
    min_coins = [float('inf')] * (V+1)
    min_coins[0] = 0
    for v in range(1, V+1):
        for i in range(len(coins)):
            if coins[i] <= v:
                min_coins[v] = min(min_coins[v], 1 + min_coins[v-coins[i]])
    output = reconstruct(coins, min_coins, V)
    return (min_coins[V], output)


def reconstruct(coins, min_coins, V):
    if min_coins[V] == float('inf'):
        raise Exception
    optimal_coins = []
    while V > 0:
        for i in range(len(coins)):
            if (coins[i] <= V) and (min_coins[V] == 1 + min_coins[V - coins[i]]):
                optimal_coins.append(coins[i])
                V = V - coins[i]
    return optimal_coins