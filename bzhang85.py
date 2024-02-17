import streamlit as st
from typing import Tuple, List, Dict, Callable
from copy import deepcopy

# Constants and Map

COSTS = {'🌾': 1, '🌲': 3, '⛰': 5, '🐊': 7}
MOVES = [(0,-1), (1,0), (0,1), (-1,0)]

min_coordinate_value = 0
max_coordinate_value = 26

full_world = [
['🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌋', '🌋', '🌋', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '⛰', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰'],
['🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🐊', '🐊', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '🌾'],
['🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌲', '🌲', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '⛰', '🌾'],
['🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '⛰', '🌾', '🌾'],
['🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾'],
['🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌾', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🐊', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '⛰', '⛰', '🌋', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🌾', '🌾', '⛰', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '⛰', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌲', '🌲', '⛰', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌋', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '⛰', '⛰', '⛰', '⛰', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '⛰', '⛰', '🌾', '🐊', '🌾', '⛰', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '🌾', '🌾', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🌲', '🌲', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '⛰', '⛰', '🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊'],
['⛰', '🌋', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '⛰', '⛰', '🌋', '🌋', '🌾', '🌋', '🌋', '⛰', '⛰', '🐊', '🐊', '🐊', '🐊'],
['⛰', '🌋', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌋', '🌋', '⛰', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🐊', '🐊', '🐊', '🐊'],
['⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾']
]

# functions

def heuristic(current: Tuple[int, int], goal: Tuple[int, int]) -> int:
    return abs(goal[0] - current[0]) + abs(goal[1] - current[1])

def successors(current: Tuple[int, int], world: List[List[str]], moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    successors = []
    x, y = current
    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy
        # Check boundaries and impassable terrain
        if 0 <= new_x < len(world[0]) and 0 <= new_y < len(world) and world[new_y][new_x] != '🌋':
            successors.append((new_x, new_y))
    return successors

def calculate_actual_cost(path: List[Tuple[int, int]], world: List[List[str]], costs: Dict[str, int]) -> int:
    total_cost = 0
    for x, y in path:
        # Check for bounds
        if 0 <= x < len(world[0]) and 0 <= y < len(world):
            terrain = world[y][x]
            total_cost += costs.get(terrain, float('inf'))
        else:
            # If out of bounds, return infinite cost
            return float('inf')
    return total_cost

def insert_into_queue(queue: List[Tuple[int, Tuple[int, int], List[Tuple[int, int]]]], 
                      node: Tuple[int, int], 
                      path: List[Tuple[int, int]], 
                      cost: int) -> None:
    queue.append((cost, node, path))
    queue.sort(key=lambda x: x[0])  # Sort by total cost

def convert_path_to_offsets(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    offsets = []
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        offsets.append((dx, dy))
    return offsets

def a_star_search(world: List[List[str]], 
                  start: Tuple[int, int], 
                  goal: Tuple[int, int], 
                  costs: Dict[str, int], 
                  moves: List[Tuple[int, int]], 
                  heuristic: Callable[[Tuple[int, int], Tuple[int, int]], int]) -> List[Tuple[int, int]]:
    queue = []
    insert_into_queue(queue, start, [start], 0)
    visited = set()
    while queue:
        current_cost, current_node, path = queue.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)
        if current_node == goal:
            return convert_path_to_offsets(path) # return the offset if the goal is found
        for move in moves:
            successor = (current_node[0] + move[0], current_node[1] + move[1])
            if 0 <= successor[0] < len(world[0]) and 0 <= successor[1] < len(world) and world[successor[1]][successor[0]] != '🌋':
                successor_path = path + [successor]
                g = calculate_actual_cost(successor_path, world, costs)
                f = g + heuristic(successor, goal)
                insert_into_queue(queue, successor, successor_path, f)
    return [] # return an empty list if not founded

def pretty_print_path(world: List[List[str]], offsets: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    action_symbols = {
        (1, 0): '⏩',  # Right
        (-1, 0): '⏪', # Left
        (0, 1): '⏬',  # Down
        (0, -1): '⏫'  # Up
    }
    world_copy = deepcopy(world)
    x, y = start
    total_cost = 0

    for dx, dy in offsets:
        terrain = world_copy[y][x]
        total_cost += costs.get(terrain, float('inf'))
        world_copy[y][x] = action_symbols.get((dx, dy), '?')
        x += dx
        y += dy

    world_copy[y][x] = '🎁'

    output_string = ""
    for row in world_copy:
        output_string += "".join(row) + "<br>"

    st.markdown(output_string, unsafe_allow_html=True)
    
    return total_cost

# Streamlit Interface
st.title("A* Pathfinding Algorithm")
st.write("""
### How to Use
1. **Set Start and Goal**: Use the sidebar to select the start and goal coordinates for the pathfinding. Coordinates range from 0 to 26 for both X and Y.
2. **Find Path**: Click the 'Find Path' button in the sidebar after setting the coordinates. 
3. **View Results**: The optimal path from start to goal, along with its total cost and a visual representation, will be displayed below.
4. **Terrain Costs**: plain: 1, forest: 3, mountain: 5, swamp: 7

Feel free to experiment with different start and goal coordinates to explore various paths and terrains!
""")
# Sidebar for input
st.sidebar.header("Start and Goal Coordinates")
start_x = st.sidebar.number_input("Start X Coordinate", min_value=min_coordinate_value, max_value=max_coordinate_value, value=min_coordinate_value)
start_y = st.sidebar.number_input("Start Y Coordinate", min_value=min_coordinate_value, max_value=max_coordinate_value, value=min_coordinate_value)
goal_x = st.sidebar.number_input("Goal X Coordinate", min_value=min_coordinate_value, max_value=max_coordinate_value, value=min_coordinate_value)
goal_y = st.sidebar.number_input("Goal Y Coordinate", min_value=min_coordinate_value, max_value=max_coordinate_value, value=min_coordinate_value)

if st.sidebar.button("Find Path"):
    # Execute search
    start = (start_x, start_y)
    goal = (goal_x, goal_y)
    path = a_star_search(full_world, start, goal, COSTS, MOVES, heuristic)
    cost = pretty_print_path(full_world, path, start, goal, COSTS)

    # Display results
    st.write(f"Path: {path}")
    st.write(f"Total Cost: {cost}")


