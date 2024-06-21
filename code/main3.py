from collections import deque
from typing import Callable, List, Tuple

def get_word_neighbours(state: List[str], word_db: List[str], suffix: int = 4) -> List[List[str]]:
    """
    Returns all possible states reachable from the current state at the word level.

    :param state: Current state (list of words)
    :param word_db: List of words available for insertion
    :param suffix: Maximum number of words to operate on from the end of the state
    :return: List of all possible next states
    """

    neighbours = []
    l = len(state)
    m = min(suffix, l)

    # Generate all possible next states
    for i in range(l - m, l):
        # Erase operation
        neighbours.append(state[:i] + state[i+1:])
        
        # Insert operation
        for word in word_db:
            neighbours.append(state[:i] + [word] + state[i:])
    
    # Handle insertion at the end of the list
    for word in word_db:
        neighbours.append(state + [word])
    
    return neighbours

def bfs_edit(source: str,
             goal: Callable[[str], bool],
             word_db: List[str],
             max_depth: int,
             k: int,
             suffix: int = 4) -> List[Tuple[List[str], int]]:
    """
    Breadth-first search to find the shortest paths from source to goal states at the word level.

    :param source: Initial state (string)
    :param goal: Goal state function
    :param word_db: List of words available for insertion
    :param max_depth: Maximum depth to search
    :param k: Number of goal states to find
    :param suffix: Maximum number of words to operate on from the end of the state
    :return: List of tuples with the goal states and the costs (depths) to reach them
    """

    # Tokenize the source and target strings into lists of words
    source_words = source.split()

    # Initial state
    queue = deque([(source_words, 0)])
    visited = set([tuple(source_words)])
    goal_states = []
    
    i = 0
    while queue:
        state, cost = queue.popleft()
        i += 1

        if cost >= max_depth:
            continue

        if i == 10000:
            i = 0
            # Print current state for debugging
            print(f"Current: {' '.join(state)}, Cost: {cost}")
        
        if goal(state):
            goal_states.append((state, cost))
            if len(goal_states) >= k:
                return goal_states
        
        reachable = get_word_neighbours(state, word_db, suffix)
        for next_state in reachable:
            next_state_tuple = tuple(next_state)
            if next_state_tuple not in visited:
                visited.add(next_state_tuple)
                queue.append((next_state, cost + 1))
    
    return goal_states

# Example usage
source = "the dog ran after the"
targets = ["the dog chased the",
           "the dog barked at the",
           "the cat ran after a",
           "the cat chased a",
           "the cat barked at the"]

word_db = ["the", "dog", "chased", "ran", "after", "cat", "a", "barked", "at"]

goal_states = bfs_edit(source = source,
                       goal = lambda state: state in [target.split() for target in targets],
                       word_db = word_db,
                       max_depth = 5,
                       k = 3)
print("Goal states and their costs:")
for state, cost in goal_states:
    print(f"State: {' '.join(state)}, Cost: {cost}")

# Sampling based on cost
import random


def compute_probabilities(goal_states: List[Tuple[List[str], int]],
                          temperature: float = 1.0) -> List[float]:

    # use softmax
    import math
    costs = [cost for _, cost in goal_states]
    max_cost = max(costs)
    costs = [max_cost - cost for cost in costs]
    exp_costs = [math.exp(cost / temperature) for cost in costs]
    total = sum(exp_costs)
    probs = [exp_cost / total for exp_cost in exp_costs]
    return probs

def sample_states(distr: List[Tuple[List[str], float]]) -> List[str]:
    values = [state for state, _ in distr]
    weights = [weight for _, weight in distr]
    chosen_state = random.choices(values, weights, k=1)
    return chosen_state[0]

probs = compute_probabilities(goal_states, temperature = 0.5)
print("Probabilities:", probs)
states = [state for state, _ in goal_states]
distr = list(zip(states, probs))
print("Distribution:", distr)
print("Sampled state:", sample_states(distr))
