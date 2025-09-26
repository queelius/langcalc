from collections import deque
from typing import Callable, List, Tuple
import string

def get_neighbours(
        state: str) -> List[str]:
    """
    Returns all possible states reachable from the current state.

    We only operate on the last 5 characters of the string, to keep the search space small.

    :param state: Current state (string)
    :return: List of all possible next states
    """

    # All possible actions: (label, cost)
    neighbours = []

    l = len(state)
    m = min(3, l)

    # Generate all possible next states
    for i in range(l - m, l):
        # Erase operation
        neighbours.append(state[:i] + state[i+1:])
        
        # Insert operation
        for char in string.ascii_lowercase:
            neighbours.append(state[:i] + char + state[i:])
    
    # Handle insertion at the end of the string
    for char in string.ascii_lowercase:
        neighbours.append(state + char)
    
    return neighbours

def bfs_edit(source: str,
             goal: Callable[[str], bool],
             neighbours: Callable[[str], List[str]],
             max_depth: int) -> Tuple[str, int]:
    """
    Breadth-first search to find the shortest path from source to a goal state.

    :param source: Initial state
    :param goal: Function to check if a state is the goal
    :param neighbours: Function to generate all possible next states
    :param max_depth: Maximum depth to search
    :return: Tuple with the goal state and the cost (depth) to reach it
    """

    # Initial state
    queue = deque([(source, 0)])
    visited = set([source])
    
    i = 0
    while queue:
        state, cost = queue.popleft()
        i += 1

        if cost >= max_depth:
            continue

        if i == 10000:
            i = 0
            # Print current state for debugging
            print(f"Current: {state}, Cost: {cost}")
        
        if goal(state):
            return state, cost
        
        reachable = neighbours(state)
        for next_state in reachable:
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, cost + 1))
        
    return None, float('inf')  # In case no transformation is found

# Example usage
source = "dog ran after the"
target = "dog ran after a"

# Define the goal state
def goal(state: str) -> bool:
    # we want a state that is equal to the target
    return state == target

state, cost = bfs_edit(source = source, goal = goal, neighbours = get_neighbours, max_depth = 4)
print("Final state:", state)
print("Total cost:", cost)
