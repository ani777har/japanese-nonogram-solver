
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from itertools import product

def visualize_nonogram(grid_size, solving_steps, row_clues, col_clues):
    num_rows, num_cols = grid_size
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjusted figure size for better clue visibility

    # Initialize grid matrix with lighter gray (unknown state)
    grid_matrix = np.full((num_rows, num_cols), 0.7)  # Lighter gray for initial state

    # Plot the grid (initialize with lighter gray)
    img = ax.imshow(grid_matrix, cmap="gray", vmin=0, vmax=1)

    # Add grid lines and remove tick labels
    ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Function to draw the clues next to the rows and columns
    def draw_clues():
        # Draw row clues (aligned to the right)
        for i, row_clue in enumerate(row_clues):
            ax.text(-0.8, i, ' '.join(map(str, row_clue)), ha='right', va='center', fontsize=14, color='black')

        # Draw column clues
        for j, col_clue in enumerate(col_clues):
            for k, clue in enumerate(col_clue):
                ax.text(j, num_rows + 0.2 + k*0.8, str(clue), ha='center', va='center', fontsize=14, color='black')

    # Draw clues initially
    draw_clues()

    # Update function to animate solving steps
    def update(frame):
        grid_matrix[:, :] = 0.7  # Reset to lighter gray (unknown state)
        for i, j in product(range(num_rows), range(num_cols)):
            value = solving_steps[frame][i][j]
            if value == 1:
                grid_matrix[i, j] = 0  # Black for filled cells
            elif value == 0:
                grid_matrix[i, j] = 1  # White for empty cells
        img.set_data(grid_matrix)

    # Adjust layout to ensure clues are visible outside the grid
    plt.subplots_adjust(left=0.3, right=0.85, top=0.8, bottom=0.3)  # Adjust space around the grid

    ani = animation.FuncAnimation(
        fig, update, frames=len(solving_steps), interval=500, repeat=False
    )

    plt.show()


def solve_nonogram_no_backtracking(grid_size, row_clues, col_clues):
    num_rows, num_cols = grid_size
    grid = [[set([0, 1]) for _ in range(num_cols)] for _ in range(num_rows)]

    # List to store solving steps
    solving_steps = []

    # Precompute valid row and column domains
    row_domains = [generate_line_domains(num_cols, clue) for clue in row_clues]
    col_domains = [generate_line_domains(num_rows, clue) for clue in col_clues]

    # Enforce arc consistency
    while enforce_arc_consistency(grid, row_domains, col_domains, solving_steps):
        pass

    # Check if the grid has a unique solution
    unsolved_cells = any(len(cell) > 1 for row in grid for cell in row)
    if unsolved_cells:
        print("No unique solution found.")
        return None  # Return None to indicate no solution was found

    # Convert the grid to the final solved grid (if possible)
    solved_grid = []
    for row in grid:
        solved_row = []
        for cell in row:
            if not cell:  # If the cell is empty (no valid values)
                print("No solution exists due to empty cell.")
                return None  # Return None to indicate the puzzle couldn't be solved
            solved_row.append(list(cell)[0])  # Take the single value from each set
        solved_grid.append(solved_row)

    # Visualize the solving process
    visualize_nonogram(grid_size, solving_steps, row_clues, col_clues)

    return solved_grid

def clue_to_regex(clue):
    """
    Convert the clue to a regular expression pattern.
    Each block in the clue must have '1' repeated block_size times
    and separated by at least one '0', except at boundaries.
    """
    parts = [f"1{{{c}}}" for c in clue]  # Convert each block into a '1' repeated c times
    # Join the parts with '0+' to allow for flexible spacing between blocks
    pattern = "0+".join(parts)
    # Allow leading and trailing zeros
    return f"^0*{pattern}0*$"

def generate_line_domains(length, clue):
    """
    Generate all valid binary sequences of a given length based on a row or column clue.
    Uses a regular expression to filter out invalid patterns.
    """
    # Convert the clue to a regular expression
    regex = clue_to_regex(clue)
    # Generate all possible binary sequences of the given length
    all_sequences = [''.join(seq) for seq in product("01", repeat=length)]
    # Filter sequences that match the regex
    valid_sequences = [list(map(int, seq)) for seq in all_sequences if re.fullmatch(regex, seq)]
    return valid_sequences

def choose_line_heuristic(row_domains, col_domains, assigned, stalled_lines):
    min_domain_size = float('inf')
    max_unsolved_cells = -1
    chosen_line_type = None
    chosen_index = None

    # Check rows
    for idx, domain in enumerate(row_domains):
        line_id = f"row_{idx}"
        if line_id not in assigned and line_id not in stalled_lines:
            unsolved_cells = sum(1 for cell in domain if len(cell) > 1)
            if len(domain) < min_domain_size or (
                len(domain) == min_domain_size and unsolved_cells > max_unsolved_cells
            ):
                min_domain_size = len(domain)
                max_unsolved_cells = unsolved_cells
                chosen_line_type = "row"
                chosen_index = idx

    # Check columns
    for idx, domain in enumerate(col_domains):
        line_id = f"col_{idx}"
        if line_id not in assigned and line_id not in stalled_lines:
            unsolved_cells = sum(1 for cell in domain if len(cell) > 1)
            if len(domain) < min_domain_size or (
                len(domain) == min_domain_size and unsolved_cells > max_unsolved_cells
            ):
                min_domain_size = len(domain)
                max_unsolved_cells = unsolved_cells
                chosen_line_type = "col"
                chosen_index = idx

    # If no line is chosen, mark stalled_lines and stop the process
    if chosen_line_type is None:
        stalled_lines.clear()  # Clear stalled lines
        return None, None, None  # No line to process, exit the function

    return chosen_index, row_domains[chosen_index] if chosen_line_type == "row" else col_domains[chosen_index], chosen_line_type

# Updated enforce_arc_consistency to capture the solving steps

def enforce_arc_consistency(grid, row_domains, col_domains, solving_steps):
    assigned = set()  # Track lines already processed
    stalled_lines = set()  # Track lines that didn't lead to progress
    revised = False

    while True:
        chosen_index, chosen_domain, chosen_line_type = choose_line_heuristic(
            row_domains, col_domains, assigned, stalled_lines
        )

        if chosen_index is None:  # No more lines to process
            break
        print(f"Chosen line: {chosen_line_type} {chosen_index}")

        revised_local = False
        if chosen_line_type == "row":
            row_idx = chosen_index
            for col_idx, cell in enumerate(grid[row_idx]):
                valid_values = {valid_row[col_idx] for valid_row in chosen_domain}
                original_domain = cell.copy()
                cell.intersection_update(valid_values)

                if cell != original_domain:
                    revised = revised_local = True
                    # Update column domains
                    col_domains[col_idx] = [
                        valid_col for valid_col in col_domains[col_idx]
                        if valid_col[row_idx] in cell
                    ]
            if revised_local:
                stalled_lines.clear()  # Clear stalled lines since progress was made
            assigned.add(f"row_{row_idx}")

        else:  # chosen_line_type == "col"
            col_idx = chosen_index
            for row_idx in range(len(grid)):
                cell = grid[row_idx][col_idx]
                valid_values = {valid_col[row_idx] for valid_col in chosen_domain}
                original_domain = cell.copy()
                cell.intersection_update(valid_values)

                if cell != original_domain:
                    revised = revised_local = True
                    # Update row domains
                    row_domains[row_idx] = [
                        valid_row for valid_row in row_domains[row_idx]
                        if valid_row[col_idx] in cell
                    ]
            if revised_local:
                stalled_lines.clear()  # Clear stalled lines since progress was made
            assigned.add(f"col_{col_idx}")

        if not revised_local:  # If no progress was made, mark the line as stalled
            stalled_lines.add(f"{chosen_line_type}_{chosen_index}")

        # Record the grid state at this step
        solving_steps.append([[max(cell) if len(cell) == 1 else None for cell in row] for row in grid])

    return revised


if __name__ == "__main__":

    # row_clues = [
    #     [9], [1, 1, 1, 1, 1, 1], [2, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 1], [1, 4, 4, 1],
    #     [1, 5, 5, 1], [1, 1, 3, 3, 1, 1], [1, 2, 2, 1], [1, 1, 1, 1, 1], [1, 3, 3, 1],
    #     [1, 11, 1], [1, 9, 1], [2, 7, 2], [1, 1, 1, 1, 1, 1], [9]
    # ]
    #
    # col_clues = [
    #     [9], [1, 1], [6, 2, 2], [1, 2, 4, 1], [2, 4, 6], [1, 4, 3, 1], [4, 3, 5], [1, 1, 3, 1],
    #     [4, 3, 5], [1, 4, 3, 1], [2, 4, 6], [1, 2, 4, 1], [6, 2, 2], [1, 1], [9]
    # ]

    # row_clues = [
    #     [2], [2, 3], [2, 3], [4], [4, 5], [5, 8], [2, 13], [12, 4, 1], [11, 7], [2, 6, 7], [6, 6], [3, 2, 4], [3, 4],
    #     [3, 7]
    # ]
    # col_clues = [
    #     [2], [1, 4], [3, 6], [5, 2, 1], [6, 2], [12], [3, 8], [2, 6], [1, 6],
    #     [7, 1], [8, 1], [4, 3], [3, 3, 2], [10], [9], [7, 1], [6], [3], [3]
    # ]

    # %%
    # row_clues = [
    #     [1, 6, 1], [3, 4, 3], [3, 4, 3], [4, 4], [1, 1, 1, 1], [4, 1, 1, 4], [4, 1, 1, 4], [4, 4], [4, 2, 2, 1],
    #     [5, 3, 1], [4, 4, 3], [3, 2], [3, 1, 1, 1, 1], [3, 1, 1, 1, 1]
    # ]
    # col_clues = [
    #     [1, 10], [3, 9], [3, 9], [10], [1, 1, 1, 2], [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 2, 1], [1, 1, 1, 2], [10],
    #     [3, 9], [3, 3, 2], [1, 6, 2]
    # ]

    #thanks
    # row_clues = [[0], [3, 1, 1, 3, 1, 1, 1, 1, 3], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 3, 3, 2, 1, 2, 3],
    #              [1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
    #              [1, 1, 1, 1, 1, 1, 1, 1, 1, 3], [0]]
    #
    # col_clues = [[1], [5], [1], [0], [5],
    #              [1], [5], [0], [5], [1, 1], [5], [0], [5], [1], [1], [5], [0], [5],
    #              [1], [2, 2], [0], [3, 1], [1, 1, 1], [1, 3]]

    #nota
    # row_clues = [
    #     [3], [2,1], [2,3], [1, 2,1], [2,1], [1,1],
    #     [1,3], [3,4], [4,4], [4,2], [2]
    # ]
    #
    # col_clues = [
    #     [2], [4], [4], [8], [1,1], [1,1,1], [1,1 ,2], [1,1, 4], [1,1,4],
    #     [8]
    # ]
    #beast
    # row_clues = [
    #     [4], [8], [10], [12], [12], [3, 2, 3], [4, 2, 4], [14], [14],
    #     [2, 2, 2, 2], [1, 2, 2, 2, 1], [14], [4, 4, 4], [2, 2, 2]
    # ]
    # col_clues = [
    #     [7], [7, 3], [7, 4], [8, 3], [4, 3, 1], [5, 3, 2], [9, 4], [9, 4], [5, 3, 2], [4, 3, 1], [8, 3], [7, 4], [7, 3],
    #     [7]
    # ]
    row_clues = [
        [1, 1], [1,2], [2], [5], [1]
    ]
    col_clues = [
        [1,1], [1,2], [3], [2,1], [1,1]
    ]
    m, n = len(row_clues), len(col_clues)
    grid_size = (m, n)  # Example grid size

    solved_grid = solve_nonogram_no_backtracking(grid_size, row_clues, col_clues)
    print("Solved Grid:", solved_grid)


