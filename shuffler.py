from tqdm import tqdm
import numpy as np
from numba import njit, prange
from collections import deque


@njit
def get_deck(n=52):
    return list(range(n))


def shuffle_once(input_list):
    second_half_start = (len(input_list) + 1) // 2
    first_half = input_list[:second_half_start]
    second_half = input_list[second_half_start:]

    shuffled_deque = deque()

    while len(first_half) > 0 or len(second_half) > 0:
        if len(first_half) == 0:
            choose_first_half = 0
        elif len(second_half) == 0:
            choose_first_half = 1
        else:
            choose_first_half = (np.random.rand() < 0.5)
            
        if choose_first_half:
            shuffled_deque.appendleft(first_half.pop())
        else:
            shuffled_deque.appendleft(second_half.pop())

    return list(shuffled_deque)


@njit
def shuffle_once_np(input_np_array):
    # half decks with [top, bottom) indices
    
    first_half_bottom = (len(input_np_array) + 1) // 2
    second_half_bottom = len(input_np_array)
    
    first_half_top = 0
    second_half_top = first_half_bottom

    first_half_index = first_half_bottom - 1
    second_half_index = second_half_bottom - 1

    shuffled_list = np.zeros(len(input_np_array), dtype=input_np_array.dtype)

    output_index = len(input_np_array) - 1
    while first_half_index >= 0 or second_half_index >= second_half_top:
        if first_half_index < first_half_top:
            choose_first_half = False
        elif second_half_index < second_half_top:
            choose_first_half = True
        else:
            choose_first_half = (np.random.rand() < 0.5)
            
        if choose_first_half:
            shuffled_list[output_index] = input_np_array[first_half_index]
            first_half_index -= 1
        else:
            shuffled_list[output_index] = input_np_array[second_half_index]
            second_half_index -= 1
        output_index -= 1

    return shuffled_list


def shuffle(input_list, n=1):
    for i in range(n):
        input_list = shuffle_once(input_list)
    return input_list


@njit(parallel=True)
def run_shuffling_simulation_fast(n_simulations, n_shuffles, deck_size):
    distribution_of_results = np.zeros((deck_size, deck_size), dtype=np.int64)

    for sim_i in prange(n_simulations):
        deck = np.arange(deck_size)
        for _ in range(n_shuffles):
            deck = shuffle_once_np(deck)
        for i in range(deck_size):
            distribution_of_results[deck[i], i] += 1

    return distribution_of_results


def run_shuffling_simulation_old(n_simulations, n_shuffles, deck_size=52, suppress_tqdm=True):
    distribution_of_results = np.zeros((deck_size, deck_size), dtype=np.int64)

    for sim_i in tqdm(range(n_simulations), desc="simulation", disable=suppress_tqdm, ):
        deck = get_deck(deck_size)
        shuffled_deck = shuffle(deck, n_shuffles)
        distribution_of_results[shuffled_deck, range(deck_size)] += 1

    return distribution_of_results


if __name__ == "__main__":
    n_simulations = 100
    outcomes_by_n_shuffles = {}
    for n_shuffles in range(1, 25):
        outcomes_by_n_shuffles[n_shuffles] = run_shuffling_simulation_fast(
            n_simulations, n_shuffles
        )

    np.savez(
        f"outcomes_{n_simulations}_simulations.npz",
        **{str(k): v for k, v in outcomes_by_n_shuffles.items()}
    )
