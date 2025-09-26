from collections import deque
from typing import Callable, List, Tuple

def get_word_neighbours(state: List[str], word_db: List[str], max_words: int = 5) -> List[List[str]]:
    """
    Returns all possible states reachable from the current state at the word level.

    :param state: Current state (list of words)
    :param word_db: List of words available for insertion
    :param max_words: Maximum number of words to operate on from the end of the state
    :return: List of all possible next states
    """

    neighbours = []
    l = len(state)
    m = min(max_words, l)

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
             target: str,
             word_db: List[str],
             max_depth: int) -> Tuple[List[str], int]:
    """
    Breadth-first search to find the shortest path from source to a goal state at the word level.

    :param source: Initial state (string)
    :param target: Goal state (string)
    :param word_db: List of words available for insertion
    :param max_depth: Maximum depth to search
    :return: Tuple with the goal state and the cost (depth) to reach it
    """

    # Tokenize the source and target strings into lists of words
    source_words = source.split()
    target_words = target.split()

    # Initial state
    queue = deque([(source_words, 0)])
    visited = set([tuple(source_words)])
    
    i = 0
    while queue:
        state, cost = queue.popleft()
        i += 1

        if cost >= max_depth:
            continue

        if i == 100:
            i = 0
            # Print current state for debugging
            print(f"Current: {' '.join(state)}, Cost: {cost}")
        
        if state == target_words:
            return state, cost
        
        reachable = get_word_neighbours(state, word_db)
        for next_state in reachable:
            next_state_tuple = tuple(next_state)
            if next_state_tuple not in visited:
                visited.add(next_state_tuple)
                queue.append((next_state, cost + 1))
        
    return None, float('inf')  # In case no transformation is found

# Example usage
source = "the dog ran after the"
target = "the dog chased the"
word_db = ["the", "dog", "chased", "ran", "after", "cat", "a"]

state, cost = bfs_edit(source = source, target = target, word_db = word_db, max_depth = 10)
print("Final state:", ' '.join(state) if state else "None")
print("Total cost:", cost)
