import cv2
import re
import numpy as np
from itertools import product
from copy import deepcopy
from tqdm import tqdm


def generate_regex_pattern(constraints):
    """Generate a regex pattern based on constraints."""
    regex_pattern = '0*'  # Allow for leading zeros
    for i, count in enumerate(constraints):
        regex_pattern += f"1{{{count}}}"  # Block of 1's
        if i < len(constraints) - 1:
            regex_pattern += '0+'  # One or more 0's between blocks of 1's
    regex_pattern += '0*'  # Allow for trailing zeros
    return regex_pattern


def generate_valid_combinations(length, constraints):
    """Generate all valid row/column configurations."""
    regex_pattern = re.compile(generate_regex_pattern(constraints))
    base = product([0, 1], repeat=length)
    return [list(seq) for seq in base if regex_pattern.fullmatch(''.join(map(str, seq)))]



class NonogramCSP:
    def __init__(self, m, n, row_constraints, col_constraints):
        self.m = m
        self.n = n
        self.row_constraints = row_constraints
        self.col_constraints = col_constraints
        self.grid = [[None for _ in range(n)] for _ in range(m)]
        self.domains = {'row': {}, 'col': {}}

        # Initialize domains for each row and column using constraints
        for i in range(m):
            self.domains['row'][i] = generate_valid_combinations(n, row_constraints[i])
        for j in range(n):
            self.domains['col'][j] = generate_valid_combinations(m, col_constraints[j])

    def is_consistent(self, row, col, value):
        """Check if the current assignment is consistent with row and column constraints."""
        print(f"Checking consistency for row {row}, column {col} with value: {value}")
        # Check consistency for the row
        if row is not None:
            for j in range(self.n):
                if self.grid[row][j] is not None and self.grid[row][j] != value[j]:
                    print(f"Row consistency failed at row {row}, column {j}")
                    return False

        # Check consistency for the column
        if col is not None:
            for i in range(self.m):
                if self.grid[i][col] is not None and self.grid[i][col] != value[i]:
                    print(f"Column consistency failed at row {i}, column {col}")
                    return False
        return True

    def forward_checking(self, row, col, value):
        """Apply forward checking to update domains after an assignment."""
        print(f"Applying forward checking for row {row}, col {col} with value: {value}")
        affected_rows = []
        affected_cols = []

        # Check affected rows (when a row is assigned)
        if row is not None:
            affected_rows.append(row)
            for j in range(self.n):
                # Check the intersecting columns
                if self.grid[row][j] is None:  # only affected if the cell is unassigned
                    new_domains = []
                    for domain in self.domains['col'][j]:
                        if domain[row] == value[j]:  # Compare with column values
                            new_domains.append(domain)
                    print(f"Updating domain for column {j}: {new_domains}")
                    self.domains['col'][j] = new_domains

            # Now check for consistency in columns affected by this row assignment
            for j in range(self.n):
                if self.grid[row][j] == 1:  # The row has a '1' at column j
                    # Remove all inconsistent domain values for column j
                    self.domains['col'][j] = [col_val for col_val in self.domains['col'][j] if col_val[row] == 1]
                    print(f"Updated domain for column {j} after row assignment: {self.domains['col'][j]}")
                elif self.grid[row][j] == 0:  # The row has a '0' at column j
                    # Remove all inconsistent domain values for column j where row's value isn't zero
                    self.domains['col'][j] = [col_val for col_val in self.domains['col'][j] if col_val[row] == 0]
                    print(f"Updated domain for column {j} after row assignment: {self.domains['col'][j]}")

        # Check affected columns (when a column is assigned)
        if col is not None:
            affected_cols.append(col)
            for i in range(self.m):
                # Check the intersecting rows
                if self.grid[i][col] is None:  # only affected if the cell is unassigned
                    new_domains = []
                    for domain in self.domains['row'][i]:
                        if domain[col] == value[i]:  # Compare with row values
                            new_domains.append(domain)
                    print(f"Updating domain for row {i}: {new_domains}")
                    self.domains['row'][i] = new_domains

            # Now check for consistency in rows affected by this column assignment
            for i in range(self.m):
                if self.grid[i][col] == 1:  # The column has a '1' at row i
                    # Remove all inconsistent domain values for row i
                    self.domains['row'][i] = [row_val for row_val in self.domains['row'][i] if row_val[col] == 1]
                    print(f"Updated domain for row {i} after column assignment: {self.domains['row'][i]}")
                elif self.grid[i][col] == 0:  # The column has a '0' at row i
                    # Remove all inconsistent domain values for row i where column's value isn't zero
                    self.domains['row'][i] = [row_val for row_val in self.domains['row'][i] if row_val[col] == 0]
                    print(f"Updated domain for row {i} after column assignment: {self.domains['row'][i]}")

        return affected_rows, affected_cols

    def restore_domains(self, affected_rows, affected_cols, backup_domains):
        """Restore domains after backtracking."""
        print(f"Restoring domains for affected rows: {affected_rows}, affected cols: {affected_cols}")
        for row in affected_rows:
            self.domains['row'][row] = backup_domains['row'][row]
        for col in affected_cols:
            self.domains['col'][col] = backup_domains['col'][col]

    def select_unassigned_variable(self):
        """Select the variable with the minimum remaining values (MRV) heuristic."""
        print("Selecting unassigned variable based on MRV heuristic...")
        min_domain_size = float('inf')
        var = None

        # Look for the row with the smallest domain size
        for i in range(self.m):
            if None in self.grid[i]:
                domain_size = len(self.domains['row'][i])
                print(f"Row {i} has domain size {domain_size}")
                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    var = ('row', i)

        # Look for the column with the smallest domain size
        for j in range(self.n):
            if any(self.grid[i][j] is None for i in range(self.m)):
                domain_size = len(self.domains['col'][j])
                print(f"Column {j} has domain size {domain_size}")
                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    var = ('col', j)

        print(f"Selected variable: {var}")
        return var

    def save_visualization(self, filename="nonogram_solution.mp4", frame_rate=1):
        """Save the solving process as a video."""
        height = len(self.grid)
        width = len(self.grid[0])
        cell_size = 50  # Size of each cell in pixels

        # Initialize video writer
        out = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate,
            (width * cell_size, height * cell_size)
        )

        for state in self.visualization_states:
            # Create an image for the current state
            img = np.ones((height * cell_size, width * cell_size, 3), dtype=np.uint8) * 255
            for i, row in enumerate(state):
                for j, cell in enumerate(row):
                    color = (0, 0, 0) if cell == 1 else (255, 255, 255) if cell == 0 else (200, 200, 200)
                    cv2.rectangle(
                        img,
                        (j * cell_size, i * cell_size),
                        ((j + 1) * cell_size, (i + 1) * cell_size),
                        color,
                        -1
                    )
                    cv2.rectangle(
                        img,
                        (j * cell_size, i * cell_size),
                        ((j + 1) * cell_size, (i + 1) * cell_size),
                        (100, 100, 100),
                        1
                    )
            out.write(img)

        out.release()
        
    def backtrack(self):
        """Perform backtracking with visualization and insights."""
        if not hasattr(self, "visualization_states"):
            self.visualization_states = []

        if self.is_complete():
            self.visualization_states.append([row.copy() for row in self.grid])
            return self.grid

        # Select the next variable using MRV
        var_type, idx = self.select_unassigned_variable()

        # Retrieve possible values for the selected variable
        possible_values = self.domains[var_type][idx]
        original_grid_state = [row.copy() for row in self.grid]  # Backup grid state
        backup_domains = deepcopy(self.domains)  # Deep copy to save all domain states

        for value in tqdm(possible_values):
            # Assign the value to the grid
            if var_type == 'row':
                self.grid[idx] = value
            elif var_type == 'col':
                for row in range(self.m):
                    self.grid[row][idx] = value[row]

            # Capture the current state for visualization
            self.visualization_states.append([row.copy() for row in self.grid])

            # Check if the assignment is consistent
            if self.is_consistent(idx if var_type == 'row' else None,
                                idx if var_type == 'col' else None,
                                value):
                # Apply forward checking
                affected_rows, affected_cols = self.forward_checking(
                    idx if var_type == 'row' else None,
                    idx if var_type == 'col' else None,
                    value
                )

                # Check if any neighbor's domain is reduced to zero
                if all(len(self.domains['row'][r]) > 0 for r in affected_rows) and \
                        all(len(self.domains['col'][c]) > 0 for c in affected_cols):
                    # Recur to continue the search
                    result = self.backtrack()
                    if result is not None:
                        return result

            # Restore grid and domains after trying this value
            self.grid = [row.copy() for row in original_grid_state]
            self.domains = deepcopy(backup_domains)

        # If no values work for the current variable, return None to trigger backtracking
        return None

    def is_complete(self):
        """Check if the grid is completely assigned."""
        for i in range(self.m):
            if None in self.grid[i]:
                return False
        return True
    
    def display_grid(self, state=None):
        """Display the grid in a readable format."""
        if state is None:
            state = self.grid
        print("\nCurrent Grid State:")
        for row in state:
            print(" ".join([" " if cell == 1 else "#" if cell == 0 else "." for cell in row]))
        print("\n")

    def analyze_state(self, state=None):
        """Analyze and provide insights into the current state."""
        if state is None:
            state = self.grid
        completed_rows = sum(1 for row in state if None not in row and sum(row) > 0)
        completed_cols = sum(
            1 for col in zip(*state) if None not in col and sum(col) > 0
        )
        total_cells = len(state) * len(state[0])
        filled_cells = sum(cell == 1 for row in state for cell in row if cell is not None)
        empty_cells = sum(cell == 0 for row in state for cell in row if cell is not None)

        print(f"Statistics:")
        print(f"- Completed Rows: {completed_rows}/{len(state)}")
        print(f"- Completed Columns: {completed_cols}/{len(state[0])}")
        print(f"- Filled Cells: {filled_cells}/{total_cells}")
        print(f"- Empty Cells: {empty_cells}/{total_cells}")
        print("\n")


row_constraints = [
    [6, 2], [1, 1, 2], [1, 1, 1], [1, 1, 1], [7, 1], [8, 1], [6, 1, 1], [6, 1, 1], [6, 1,1 ], [6, 1], [6], [8]
]

col_constraints = [
    [1], [12], [1, 8], [1, 8], [1, 8], [1, 8], [12], [2, 1], [1, 4], [2, 1], [8]
]
m, n = len(row_constraints), len(col_constraints)

nonogram = NonogramCSP(m, n, row_constraints, col_constraints)
solution = nonogram.backtrack()

if solution:
    nonogram.display_grid(solution)
    nonogram.analyze_state(solution)
    nonogram.save_visualization("nonogram_solution.mp4", frame_rate=2)
    print("Video saved as 'nonogram_solution.mp4'")
else:
    print("No solution found.")


#Other examples

#73718 15*15
# row_constraints = [
#     [9], [1, 1, 1, 1, 1, 1], [2, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 1], [1, 4, 4, 1],
#     [1, 5, 5, 1], [1, 1, 3, 3, 1, 1], [1, 2, 2, 1], [1, 1, 1, 1, 1], [1, 3, 3, 1],
#     [1, 11, 1], [1, 9, 1], [2, 7, 2], [1, 1, 1, 1, 1, 1], [9]
# ]
#
# col_constraints = [
#     [9], [1, 1], [6, 2, 2], [1, 2, 4, 1], [2, 4, 6], [1, 4, 3, 1], [4, 3, 5], [1, 1, 3, 1],
#     [4, 3, 5], [1, 4, 3, 1], [2, 4, 6], [1, 2, 4, 1], [6, 2, 2], [1, 1], [9]
# ]


#66259
# row_constraints = [
#     [1, 2, 1, 1, 1], [3, 6, 3], [1, 4], [1], [14], [2, 13], [2, 1, 12],
#     [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 3, 1, 1, 1, 1, 1], [1, 3, 1, 1], [17]
# ]
#
# col_constraints = [
#     [6], [2, 1], [1, 2, 1, 3], [5, 1, 4], [1, 2, 1, 3], [3, 1], [1, 8],
#     [2, 3, 1], [1, 3, 1], [1, 3, 2, 1], [1, 3, 1], [3, 3, 2, 1], [2, 3, 1],
#     [1, 3, 2, 1], [3, 3, 1], [1, 3, 1], [2, 7]
# ]

#65983
# row_constraints = [
#     [5], [8, 7], [1, 3, 11], [5, 7], [2], [6, 2, 10], [2, 3, 11], [1, 2], [8], [6]
# ]
#
# col_constraints = [
#     [2, 3], [2, 1, 2, 2], [2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 2], [2, 2, 2],
#     [2, 3], [2, 2], [1, 1], [4], [3, 1], [2, 2], [2, 2], [3, 2],
# [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [1, 1]
# ]

#65967
# row_constraints = [
#     [5, 1], [2, 2, 1], [1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1], [2, 2, 1], [5, 1],
#     [1], [1, 1, 1], [5, 1], [5, 1], [5, 1], [3, 1], [1, 2], [3]
# ]
# col_constraints = [
#     [5], [2, 2], [1, 1, 1], [1, 1, 3], [1, 1, 1, 5], [2, 2, 5], [5, 5, 1], [3, 1], [2], [14]
# ]

#72356
# row_constraints = [
#     [6, 2], [1, 1, 2], [1, 1, 1], [1, 1, 1], [7, 1], [8, 1], [6, 1, 1], [6, 1, 1], [6, 1,1 ], [6, 1], [6], [8]
# ]
#
# col_constraints = [
#     [1], [12], [1, 8], [1, 8], [1, 8], [1, 8], [12], [2, 1], [1, 4], [2, 1], [8]
# ]



#64653
# row_constraints = [
#     [4], [1, 2], [1, 2, 1, 1], [1, 1, 1, 2], [3, 4, 4], [5, 4], [2, 2, 2], [2, 1, 1, 1, 2],
#     [1, 1, 1, 1], [1, 2, 3, 1], [3, 6], [1, 3, 2], [2, 2, 1, 1], [1, 1, 1, 1]
# ]
# col_constraints = [
#     [2, 1], [1, 5], [2, 1, 2], [2, 1, 4], [2, 2, 1, 2, 1], [1, 2, 2, 1], [1, 1, 1, 2, 2], [1, 2, 1, 4],
#     [2, 2], [4, 4], [3, 2, 2], [2, 1], [4, 2], [5, 3]
# ]

#64051
# row_constraints = [
#     [4, 4], [3, 3], [5, 5], [5, 5], [2, 2], [2, 1, 1, 2], [16], [14], [10],
#     [14], [2, 8, 2], [10], [2, 4, 2]
# ]
# col_constraints = [
#     [4, 1], [7, 2], [4, 2, 1, 1], [4, 4, 2], [1, 2, 6], [1, 1, 6], [8], [7], [7],
#     [8], [1, 1, 6], [1, 2, 6], [4, 4, 2], [4, 2, 1, 1], [7, 2], [4, 1]
# ]

#64017
# row_constraints = [
#     [1, 1, 1], [1, 1, 1], [5, 1, 1], [1, 2, 2], [1, 5], [1, 2, 1, 2], [1, 5],
#     [2, 3, 1], [9, 1], [1, 3, 2], [6], [2, 2], [2, 2]
# ]
# col_constraints = [
#     [3], [1], [10], [1, 2], [3, 1, 1, 1], [4, 1, 2], [3, 6], [7], [3, 6],
#     [4, 1, 3], [1, 2, 1, 1], [2], [2]
# ]

#63223
# row_constraints = [
#     [2], [1, 2], [7], [9], [1, 1, 3], [1, 1, 1, 2], [8, 1, 1], [1, 5], [1, 1, 4], [3, 6],
#     [1, 1, 1, 3, 1], [8, 1], [1, 1, 1], [5, 1], [1, 1], [1, 1]
# ]
#
# col_constraints = [
#     [10, 1], [2, 1, 1, 1, 2], [3, 1, 3, 1], [1, 5, 1, 1, 1], [4, 1, 3, 1],
#         [3, 1, 1, 1, 1, 1], [2, 1, 4, 1], [10, 1], [2, 4, 1], [6, 1], [1, 2, 1], [2]
# ]


#74109
# row_constraints = [
#     [1, 5, 4], [4, 2, 3], [6, 3, 2], [4, 1, 1, 1], [2, 1, 1, 2], [3, 3], [2], [3, 2, 1],
#     [1, 1, 3, 2], [1, 2, 2, 3], [2, 3, 1, 2], [1, 1, 1, 1], [2, 3, 2], [3, 6], [4]
# ]
# col_constraints = [
#     [3, 3, 5], [6, 1, 3], [4, 1, 3, 2], [5, 1, 1, 1], [1, 2, 1, 2], [3, 3], [2, 1],
#     [2, 2], [3, 3, 3], [1, 1, 2, 1, 2], [1, 1, 2, 1, 1, 1], [2, 3, 3, 2], [3, 6], [4]
# ]

#74083
# row_constraints = [
#     [3, 1], [1, 1, 2], [1, 1, 1], [6, 1], [2, 4], [1, 2, 1, 1], [1, 3, 1], [2, 1], [6]
# ]
# col_constraints = [
#     [4], [4, 2], [1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 1, 1], [3, 1], [2, 1], [2, 1], [2, 1, 1], [2, 3]
# ]
#

#73867
# row_constraints = [
#     [2], [7], [1, 1], [2, 1], [1, 1], [1, 1], [1, 1], [3, 1, 3, 1], [2, 1, 2, 2, 1, 2],
#     [2, 2, 2, 2], [5, 1, 5, 1], [3, 1, 3, 1], [3, 1, 3, 1], [5, 5]
# ]
# col_constraints = [
#     [5], [7], [1, 4], [5, 1, 1], [1, 1, 1], [2, 3, 1], [1, 5], [1], [1, 5], [1, 7], [1, 1, 4],
#     [8, 1, 1], [2, 1, 1], [1, 3, 1], [5]
# ]

#73845
# row_constraints = [
#     [1], [3], [3, 1], [6], [6], [6], [1, 1], [2, 2]
# ]
# col_constraints = [
#     [2], [2, 1], [8], [3], [3], [3], [3, 1], [5], [1]
# ]

#73793
# row_constraints = [
#     [6, 2, 5], [4, 1, 2, 4], [3, 4, 1, 3], [2, 1, 3, 2], [1, 2, 1, 1],
#     [2], [5], [3, 2, 2], [3, 8], [5, 7], [19]
# ]
# col_constraints = [
#     [5, 3], [4, 3], [3, 4], [2, 1, 2], [1, 1, 2], [1, 1, 1], [1, 1], [1, 2, 1],
#     [1, 1, 3, 1], [1, 1, 3, 1], [4, 1, 1], [2, 2, 1], [1, 1, 3], [1, 1, 3], [1, 1, 3], [2, 3], [3, 3], [4, 3], [5, 3]
# ]

#72653
# row_constraints = [
#     [5], [2, 1], [1], [5], [5], [1, 3], [2, 2], [1, 2, 1], [2, 1, 2], [5], [1, 3, 1], [1], [1, 2, 1],
#     [1], [2], [1], [2], [3], [1, 2], [3, 1], [4, 1], [2], [1, 1, 2], [1, 1, 2], [3], [5], [3]
# ]
#
# col_constraints = [
#     [1, 1, 1, 1, 3, 2, 2, 1, 2], [1, 1, 3, 1, 2, 2, 1, 2], [2, 2, 1, 2, 3, 2, 1, 2],
#     [1, 2, 5, 1, 2, 1, 1], [1, 3, 1, 2, 1, 1, 2, 2], [1, 4, 2, 1, 2, 3], [2, 1, 4, 1, 1, 3, 2]
# ]

#66825
# row_constraints = [
#     [5], [3, 2], [3, 3], [2, 3, 1], [1, 1, 2], [1, 3, 1], [1, 2, 1, 4], [2, 1, 6], [1, 1, 3], [1, 5], [3, 3], [2, 5]
# ]
# col_constraints = [
#     [6], [3, 2, 1], [3, 1, 3], [2, 1, 1], [1, 2, 2, 1], [1, 1, 1], [4, 4, 1], [2, 1, 3, 1], [1, 8], [2, 2, 3], [3, 2], [2, 1]
# ]

#69309
# row_constraints = [
#     [1, 3], [3, 1, 1], [6, 1], [9], [11], [13], [1, 1], [1, 3, 1], [1, 1, 1, 4],
#     [1, 3, 1, 1], [1, 1, 2], [1, 1, 1], [1, 1, 1], [11]
# ]
# col_constraints = [
#     [1], [10], [2, 1], [3, 3, 1], [4, 1, 1, 1], [5, 3, 1], [6, 1], [5 ,1],
#     [4, 6], [6, 1, 1], [1, 3, 1, 1, 1], [14], [1]
# ]

#69263
# row_constraints = [
#     [1, 1], [1, 5, 1], [3, 1, 2], [1, 1, 1, 1, 1], [3, 1, 2, 1], [1, 1, 2, 1, 2],
#     [1, 2, 1, 1, 1, 2], [2, 2, 1, 1, 1], [6, 6], [2, 3, 2, 2], [2, 2, 2, 2]
# ]
# col_constraints = [
#     [1], [1], [2], [2, 1, 1], [4, 1, 3], [1, 1, 2], [3, 1, 1, 1, 1], [1, 1, 1, 3],
#     [3, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 4], [2, 1, 1, 2], [1, 1, 2, 1],
#     [1, 1, 3], [1, 2, 2], [1, 1]
# ]

#183 (20,10)

# row_constraints = [
#     [3], [5], [7], [7], [9], [5, 2], [6], [6], [6], [8], [9], [8], [7], [4], [3], [3, 2], [3, 1, 2], [3, 2], [8], [5]
# ]
#
# col_constraints = [
#     [3], [3,8], [6,10], [18], [14,3], [13,2], [5,7,1,2], [4,5,1,2], [4,3,5], [2,3]
# ]

#159 (7,10)
# row_constraints = [
#     [1], [3], [7, 2], [1, 7], [7], [5], [3]
# ]
# col_constraints = [
#     [2], [1,1], [4], [6], [7], [6], [4], [2], [2], [1]
# ]

#157 (10,20)

# row_constraints = [
#     [3], [5], [6], [2, 7], [6, 3], [13], [17], [16], [14], [12]
# ]
# col_constraints = [
#     [1], [2], [3], [5], [5], [6], [6], [7], [7], [6], [6], [5], [5], [9], [10], [9], [4,2], [3], [2], [1]
# ]


#80 (11,11)

# row_constraints = [
#     [5], [7], [9], [2, 1, 2], [11], [11], [2, 5, 2], [3, 3], [3, 3], [7], [5]
# ]
# col_constraints = [
#     [5], [7], [2,2,3], [3,3,3], [3, 3, 2], [7,2], [3, 3, 2], [3,3,3], [2,2,3], [7], [5]
# ]


#79 (8,11)

# row_constraints = [
#     [2], [3], [1], [3, 4], [1, 8], [10], [2, 4], [3, 1, 3]
# ]
# col_constraints = [
# [2], [1, 1], [3, 1], [7], [2,4], [2, 3], [3,1], [4], [5], [4], [2]
#  ]


#77 (14,16)

# row_constraints = [
#     [3], [5], [3, 3, 2], [3, 1, 5], [3, 3, 4], [3, 5, 3], [3, 7, 3], [3, 9, 2],
#     [2, 4, 4, 1], [6, 1, 1, 5], [4, 1, 1, 4], [4, 4], [13], [13]
# ]
# col_constraints = [
#     [3], [4], [3, 5], [3, 6], [3,7], [3, 8], [3, 4, 2], [3, 4, 2, 2], [2, 5, 2], [3, 4, 2, 2],
#     [3, 4, 2], [3, 8], [3, 7], [5, 6], [6, 5], [4]
# ]


#70 (20,9)

# row_constraints = [
#     [4], [6], [8], [6, 1], [4, 4], [4], [5], [5], [4], [4], [4], [4], [3], [3], [2], [2], [2], [2], [2,1], [4]
# ]
# col_constraints = [
#     [5, 5], [7, 8], [14, 2], [13, 1], [4, 6, 1], [5, 4, 1], [2,1,1], [3], [1]
# ]

#61 (20,10)

# row_constraints = [
#     [4], [2, 3], [7], [6], [2, 2], [1, 2], [2, 3], [1, 4], [1, 5], [1, 5], [1, 5],
#     [1, 3, 1], [1, 3, 1], [1, 2, 2], [2, 1, 3], [1, 1, 3], [2, 3], [1, 3], [6], [10]
# ]
# col_constraints = [
#     [1, 9, 1], [2, 3, 3, 1], [5, 2, 1], [1, 2, 2], [4, 8, 2], [4, 7, 2], [12, 6], [8, 7], [14], [1]
# ]


#49 (20,9)

# row_constraints = [
#     [1], [2], [3], [3], [1, 1], [1, 1], [1, 1], [2], [3], [2, 1], [2, 3],
#     [2, 2, 2], [2, 2, 2], [2, 1, 2], [2, 1, 2], [2, 1, 2], [5], [1], [1], [2]
# ]
# col_constraints = [
#     [5], [7], [2, 2], [1, 2,1,1], [20], [3,1,1,1], [5,2,2], [5], [3]]


#48 (8,20)

# row_constraints = [
#     [3], [2, 2], [1, 2], [1, 16], [1, 15], [1, 2, 4], [2, 2, 2, 1], [3, 2, 1]
# ]
# col_constraints = [
#     [6], [2,2], [1, 1], [2,2], [6], [4], [2], [2], [2], [2], [2], [2], [2], [2], [5], [5], [3], [5], [2], [1]
# ]


#45 (12,13)

# row_constraints = [
#    [3, 3], [5,5], [13], [13], [13], [13], [11], [9], [7], [5], [3], [1]
# ]
# col_constraints = [
#    [4], [6], [8], [9], [10], [10], [10], [10], [10], [9], [8], [6], [4]
# ]

#31 (6,14)

# row_constraints = [
#     [1, 5], [2, 9], [11, 2], [14], [2, 9], [1, 5]
# ]
# col_constraints = [
#     [6], [4], [2], [2], [4], [4], [6], [6], [6], [6], [6], [1, 2], [4], [2]
# ]



#15 (11,17)

# row_constraints = [
#     [6], [2, 4], [1, 6, 1, 1], [1, 7, 4], [1, 10, 2], [1, 12], [1, 12],
#     [1, 2, 5], [1, 1, 3, 8], [1, 3], [3, 5]
# ]
# col_constraints = [
#     [6], [2, 2], [1], [1,3,4], [8,1], [7,1,1], [7,1,1], [6,1], [5], [6], [6], [9], [8],
#     [2,4,1], [4,1,1], [2,1,1], [1]
# ]

#2869 (10, 10)

# row_constraints = [
#     [2, 4], [2, 2, 1], [7], [6], [1, 1, 3], [3, 3, 1], [1, 4], [1, 2], [2, 1], [5]
# ]
# col_constraints = [
#     [3,2], [3, 1, 1], [4, 1], [3,2], [4,5], [1, 4,1], [1, 5], [2,2,1], [2], [4]
# ]


#2766 (7, 20)

# row_constraints = [
#     [2], [13], [2, 2, 1], [7, 7, 1], [7, 1], [2, 13], [2]
# ]
# col_constraints = [
#     [4], [4], [2], [2], [2], [7], [7], [1,1], [1,1], [1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1],[1,1,1], [1,1,1], [1,1], [1,1], [1]
# ]

#2766 (14, 9)

# row_constraints = [
#     [4], [3, 1], [3], [2], [2], [2], [5], [6], [7], [7], [1], [1], [1], [2]
# ]
# col_constraints = [
#     [1], [2,4], [3, 6], [1,3,4,1], [4,8], [4], [3], [2], [1]
# ]

#2549 (11, 7)

# row_constraints = [
#     [1], [1], [4], [1, 1], [3], [2], [5], [3, 1], [3], [1, 4], [6]
# ]
# col_constraints = [
#     [1], [1, 1], [1, 5], [11], [1,7], [2, 1, 2], [2,1]
# ]


#2132 (18, 11)

# row_constraints = [
#     [1, 1, 1], [1, 3, 1], [7], [7], [7], [5], [3], [1], [1], [1], [2, 1, 2], [3, 1, 3], [9], [7], [5], [3] ,[1], [1]
# ]
# col_constraints = [
#     [1], [3], [5,3], [4,4], [6,4], [18], [6,4], [4,4], [5,3], [3], [1]
# ]


#2117 (10, 9)

# row_constraints = [
#     [1], [3], [2, 2], [2, 2], [2, 2], [1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 1], [9]
# ]
# col_constraints = [
#     [6], [2,1], [2,4], [2,4], [2,1], [2,2,1], [2,2,1], [2,1], [6]
# ]


#1862 (8, 10)

# row_constraints = [
#     [6], [2, 1, 1], [1, 1, 1], [9], [9], [10], [2, 2], [2, 2]
# ]
# col_constraints = [
#     [1], [3], [5], [7], [2,3], [1,3], [6], [1,5], [1,5], [6]
# ]


# 73862  (14x15)
#%%
# row_constraints = [
#    [8,3], [7,4,2], [3,1,4,1,1], [1,1,3,2,1], [1,7], [2,5], [4,2], [2,5,1,2], [1,3,5], [1,1,4,1], [1,2,5,1], [2,1,1,2,1], [3,1,3,2], [5,4]
# ]
# col_constraints = [
#    [2,7], [3,1,3], [4,2], [3,1,1], [2,1,2,1], [9], [2,4,1], [1,3,3,1], [4,2,2,1],
#    [5,5], [2,7,1], [1,4,4,1], [1,4,2,2,1], [2,2,3,2], [4,4]
# ]

# 73770  16x12
#%%
# row_constraints = [
#    [3], [5], [3,2], [7], [5], [3], [3], [6], [5,2], [7,2], [3,2],
#    [3,3], [7], [4], [1,1], [4]
# ]
# col_constraints = [
#    [1], [2], [3], [5], [2,3], [4,3,4], [6,3,2,1], [9,4], [2,5,2,1], [4,7], [2,5], [1]
# ]

# 73842 16x14
#%%
# row_constraints = [
#    [10], [12], [3,3], [2,1,2,1,2], [1,2,3], [3,5,2], [7], [3,4,1], [1,4], [2,5],
#    [3,2], [2,1], [2,1], [4], [2], [2]
# ]
# col_constraints = [
#    [2], [3,1], [3,4], [2,1,5], [2,2,3], [2,1,2,4], [2,5,2,4],
#    [2,6,3], [2,6,2], [2,2,5], [2,2,2], [3,2,1], [3,1], [2]
# ]


# 73788 17x17
# row_constraints = [
#    [7], [9], [9], [13], [12], [11], [6,1], [4,2,1], [2,1,2], [1,1,1], [1,2,3], [2,3,2,1], [1,5,2], [5,3], [5,3], [4,2], [2,2]
# ]
# col_constraints = [
#    [1], [1], [2], [8], [6,1,1,4], [7,5], [7,3], [7,5], [6,5],
#    [6,1,3], [6,3,1], [6,2], [5,4], [6, 1, 3], [2,5], [1,4], [1,2]
# ]

# 72956 20x17
#%%
# row_constraints = [
#    [6,2,4], [3,2,1,1,4], [3,2,1,1,1], [4,1,2,2,2], [3,1,1,2], [1,1,1,1], [1,1,2,3], [1,4,1,2,1], [1,1,3,4], [1,2,2,3], [1,1,1,2,1,3], [1,3,1,2,4], [1,1,1,5], [2,8,3], [3,3,2], [6,1,1], [1,4,2,1,2], [2,4,4], [2,2,1,4], [5,4]
# ]
# col_constraints = [
#    [4,9], [4,1,1,3,2], [6,1,1,1,3,2], [2,1,1,1,2,1], [1,1,1,1,1,1,2,1], [3,3,1,4,1], [2,3,1,2,2,2], [1,2,1,3,2], [2,1,1,1,1,2], [5,2,1,1,2], [1,4,2,1], [1,1,2,1,1], [1,1,1,2,1,1], [4,3,3,4], [2,2,6,3], [2,1,1,5,4], [4,5,4]
# ]

# 10040 14x14
#%%
# row_constraints = [
#    [4], [8], [10], [12], [12], [3,2,3], [4,2,4], [14], [14],
#    [2,2,2,2], [1,2,2,2,1], [14], [4,4,4], [2,2,2]
# ]
# col_constraints = [
#    [7], [7,3], [7,4], [8,3], [4,3,1], [5,3,2], [9,4], [9,4], [5,3,2], [4,3,1], [8,3], [7,4], [7,3], [7]
# ]

# 8824 13x13
#%%
# row_constraints = [
#    [5], [7], [2,1,6], [3,1,5], [1,5], [2,5], [1,7], [3,1,5], [1,1,5], [2,4], [7], [2,2], [2,2]
# ]
# col_constraints = [
#    [2], [3], [1,1,2], [2,1,1], [2,1,4], [3,2,1], [4,2,3], [7,2],
#    [9,1,1], [12], [10], [7], [4]
# ]


# 8612 11x11
#%%
# row_constraints = [
#    [5], [9], [2,1,1,1,2], [1,1,1,1,1], [11], [1], [1], [1], [1], [1,1], [3]
# ]
# col_constraints = [
#    [3], [2,1], [1, 2], [3,1], [2,1], [11], [2,1,1], [3,1,2], [1,2], [2,1], [3]
# ]

# 22501 15x14
#%%
# row_constraints = [
#    [5], [4,2], [1,4,1], [1,4,4], [4,1,1,1], [5,4,1], [1,2,2,2,1],
#    [1,2,3], [4,3], [1,1,3], [1,1,3], [1,1,4], [1,7], [2,1], [5]
# ]
# col_constraints = [
#    [5], [1,2,1], [7,3], [7,1,2] ,[4,1,3,1], [1,3,2,1,1], [2,2,1,1], [6,1,1], [1,1,1,1], [1,1,1,1], [1,2,2,1], [1,8], [1,6], [6]
# ]

# 21054 14x12
#%%
# row_constraints = [
#    [1,1,1], [4,1,1], [1,1,1,1,1], [1,1,1,2], [1,1,3,2], [8,1], [1,1,4,1],[4,3], [4,2,1], [2,1,1], [3,1,1], [2,1], [2], [1]
# ]
# col_constraints = [
#    [2], [1,1,1,2], [1,1,1,1,2], [2,3,3], [2,1,1,5], [1,5], [1,6,1], [1,6], [1,2,2,1,1], [1,1,1,1,1], [1,1,3], [2,1]
#
# ]

# 23176 11x20
#%%
# row_constraints = [
#    [1,3], [1,2,1,1], [4,1,3], [1,3,2,1], [6,5], [2,12], [1,1,2,12], [6,10, 1], [17], [1,2,1,2], [1,2,1,2]
# ]
# col_constraints = [
#    [1], [1], [2], [3], [1,2], [4,1], [2,4], [1,2,1,1], [10], [8], [9], [1,5], [1,4], [1,1,4,1], [1,5], [1,5,1], [9], [9], [1, 3, 1], [2,2]
# ]

# 22867 14x14
#%%
# row_constraints = [
#    [1,6,1], [3,4,3], [3,4,3], [4,4], [1,1,1,1], [4,1,1,4], [4,1,1,4], [4,4], [4,2,2,1], [5,3,1], [4,4,3], [3,2], [3,1,1,1,1], [3,1,1,1,1]
# ]
# col_constraints = [
#    [1,10], [3,9], [3,9], [10], [1,1,1,2], [3,2,1], [3,1,1], [3,1,1], [3,2,1], [1,1,1,2], [10], [3,9], [3,3,2], [1,6,2]
# ]

# 25992 15x14
#%%
# row_constraints = [
#    [3], [1,2], [7], [1,2], [2], [2], [2,4], [3,3,1], [6,3], [4,1,2,1], [3,2,1], [2,1,1], [3,2], [3,2], [9]
# ]
# col_constraints = [
#    [1,4], [1,6], [4,4,3], [1,1,4,2], [3,5,1], [5,2,1], [2,2,1], [1,1], [2,1], [1,2,1], [1,2,2], [4,2], [1,2], [3]
# ]

# 27390 15x10
#%%
# row_constraints = [
#    [8], [10], [4,5], [1,4], [1,1,3], [1,3], [2,4], [7], [5,1], [1,3], [3,2,3], [2,2,3], [1,2,1,2], [2,1,1], [2,1,2]
# ]
# col_constraints = [
#    [2,1,3,1], [4,1,2,2,2], [3,1,3,1,2], [3,4,2], [2,1,2,2,1],
#    [3,2,2,1,1], [4,2,1,1], [12], [7,3,1], [5,3,1]
# ]

# 28290 20x10
# row_constraints = [
#    [2,1,2], [1,1,1,1], [1,1], [5], [3,1], [4,2], [3,1], [2,1],
#    [5], [3,1] ,[2,1,3], [2,1,3,1], [1,1,2], [7], [2,1], [1,1], [2], [1], [1], [1]
# ]
# col_constraints = [
#    [2,8,2], [1,10,2], [1,4,2, 1], [1,3,1,1,1], [1,1,2,2], [1,4,1,2,4], [2,2,1,1,4], [4], [1,1], [2]
# ]

# 28999 15x9
#%%
# row_constraints = [
#    [1,1], [1,1], [1,1], [1,1], [1,1], [3,3], [3,3], [4,4], [7], [5], [0], [5], [5], [3], [1]
# ]
# col_constraints = [
#    [3], [8], [1,5,2], [3,3], [2,4], [3,3], [1,5,2], [8], [3]
# ]

# 28860 15x14
# row_constraints = [
#    [1], [2,1], [4,4], [5,5], [5,2,3], [5,6], [1,3,6], [2,6], [2,6], [12], [11], [9], [10], [9], [10]
# ]
# col_constraints = [
#    [5,4], [5,6], [14], [6,6], [4,7], [8], [9], [10], [15], [9,3], [2,3,2,1], [5,1,1], [3], [1]
# ]

# 28799 11x15
#%%
# row_constraints = [
#    [15], [1,12], [6,3], [2,2,2], [3,1,1,2,1], [1,2,1,1,2], [3,1,1,4], [5,2,5], [1,1,2,1,1], [1,1], [15]
# ]
# col_constraints = [
#    [2,1,3], [1,1,1], [1,2,1], [3,4,1], [8,1], [5,2,1], [3,2,1],
#    [5,2,1], [4,3,1], [2,2,1], [4,3,1], [8,1], [3,4,1], [2,2,1],
#    [2,1,4]
# ]

# 28688 12x14
#%%
# row_constraints = [
#    [3], [3,1,4], [1,1,1,1,1], [1,1,1,1,3], [1,1,1,1], [7,2,1],
#    [1,2,3], [1,1], [1,2], [1,4,3], [1,1,1,1,1], [1,1,1,1,1]
# ]
# col_constraints = [
#    [10], [1,1], [1,1,1,2], [2,1,1,1], [1,1], [1,3], [5], [1,1,2],
#    [1,2,1,1], [6,3], [1,3,1], [1,1,1], [3,1], [3]
# ]

# 69300 14x19
# row_constraints = [
#    [2], [2,3], [2,3],[4], [4,5], [5,8], [2,13], [12,4,1], [11,7], [2,6,7], [6,6], [3,2,4], [3,4], [3,7]
# ]
# col_constraints = [
#    [2], [1,4], [3,6] ,[5,2,1], [6,2], [12], [3,8], [2,6], [1,6] ,
#    [7,1], [8,1], [4,3], [3,3,2], [10], [9], [7,1], [6], [3], [3]
# ]

#6846 (17,11)

# row_constraints = [
#     [1], [5], [4, 1], [3,4], [1,1,3], [2,3], [4,1], [4], [6], [2,5], [1,6], [2,7], [10], [10], [8], [6], [2]
# ]
#
# col_constraints = [
#     [2], [2,3], [3,6], [2,2,4], [2,3,6], [2,14], [3,12], [2,10], [3,8], [2,6],[2,3]
# ]


#6705 (5,5)

# row_constraints = [
#     [2], [1,1], [3], [1,1], [3]
# ]
#
# col_constraints = [
#     [3], [2,1], [1,3], [1], [1]
# ]

#6557 (5,5)

# row_constraints = [
#     [2], [2], [4], [4], [4]
# ]
#
# col_constraints = [
#     [1], [1,3], [1,3], [4], [2,1]
# ]

#6556 (5,5)

# row_constraints = [
#     [1,1], [1,1,1], [1], [5], [3]
# ]
#
# col_constraints = [
#     [1,1], [1,2], [4], [1,2], [1,1]
# ]

#6555 (17,11)

# row_constraints = [
#     [2], [2,1], [4], [2], [1,1]
# ]
#
# col_constraints = [
#     [2,1], [4], [1,2], [1,1], [1]
# ]

#6554 (5,5)

# row_constraints = [
#     [1], [2], [1,1,1], [1,2], [2,2]
# ]
#
# col_constraints = [
#     [3], [1,1], [3], [2], [3]
# ]

#6552 (10,10)

# row_constraints = [
#     [2], [1,1], [1,1], [2], [1,2,1],[2,2,2], [3,2,3], [8], [6], [2]
# ]
#
# col_constraints = [
#     [2], [4], [3], [2,2], [1,7], [1,7], [2,2], [3], [4], [2,]
# ]

#6551 (5,5)

# row_constraints = [
#     [1], [3,1], [4], [1,1], [1,1]
# ]
#
# col_constraints = [
#     [4], [3], [4], [1], [1]
# ]

#6550 (10,9)

# row_constraints = [
#     [7], [1,1], [1,2, 2,1], [1,2,2,1], [1,1],[1,1,1], [1,1], [1,1,1,1], [1,1,1,1], [7]
# ]
#
# col_constraints = [
#     [5], [1,4], [1,2,1], [1,2,3], [1,1,1], [1,2,3], [1,2,1], [1,4], [5]
# ]

#6548 (10,10)

# row_constraints = [
#     [4], [4,2], [3,3], [6], [4], [5], [7], [3], [2], [1]
# ]
#
# col_constraints = [
#     [1], [1,2], [2,3], [4,5], [7], [6], [4], [5], [3], [1]
# ]

#6547 (10,10)

# row_constraints = [
#    [2,2], [2,4], [2,3,1], [1], [1,1,3],[3,3], [3,1,2], [2,1,1], [3,1], [1]
# ]
#
#
# col_constraints = [
#    [2], [2,2], [4,2], [1,1,1], [1,1,2], [2,3], [3,1], [2,2,3], [1,3], [1,3]
# ]

#6542 (5,5)

# row_constraints = [
#     [2], [1,2], [4], [1,1], [1,1]
# ]
#
# col_constraints = [
#     [2], [3], [1], [5], [2]
# ]

#6538 (5,5)

# row_constraints = [
#     [1], [2,2], [3], [3], [1]
# ]
#
# col_constraints = [
#     [1], [4], [3], [3], [1]
# ]

#6536 (10,10)

# row_constraints = [
#     [1,2], [2,3], [2,1,1], [2,3,1], [7], [3,3], [1,4], [5], [4], [2]
# ]
#
# col_constraints = [
#     [3], [6], [3,1], [3,2], [2,3], [7], [4,1], [8], [2], [2]
# ]

#6510 (10,10)

# row_constraints = [
#     [1,4,1], [4,4], [1,1], [3,3], [2,2], [1,2,1], [1,1], [2,4,2], [2,2], [6]
# ]
#
# col_constraints = [
#     [2,4], [1,2,2], [3,2], [2,1,1,1], [1,1,1,1], [1,1,1,1], [2,1,1,1], [3,2], [1,2,2], [2,4]
# ]

#6501 (10,10)

# row_constraints = [
#     [4], [1,4], [1,2,1], [1,2,2,1], [1,2,1], [1,2,1], [1,2,2,1], [1,2,1], [1,4,1], [1,4]
# ]
#
# col_constraints = [
#     [1,1,1,1], [1,1,1,1,1], [1,1], [1,2,2,1], [1,2,2,1], [1,1], [1,2,2,1], [1,2,2,1], [1,1], [9]
# ]

#6499 (10,8)

# row_constraints = [
#     [2], [4], [3], [4], [4], [3], [3], [1,1], [2,2], [6]
# ]
#
# col_constraints = [
#     [2], [2,2], [4,1], [5,1], [7,1], [2,1,1], [1,1,2], [2]
# ]

#6498 (10,9)

# row_constraints = [
#     [6], [5,2], [6], [3], [6], [6], [5,3], [4,3], [2,2] , [1,1]
# ]
#
# col_constraints = [
#     [1,1], [2,2], [3,2], [3,5], [10], [6], [6], [2,5], [1,6]
# ]

#6495 (8,10)

# row_constraints = [
#     [1,3], [3,1], [1,5,2], [1,7], [10], [6,2], [3,1], [3]
# ]
#
# col_constraints = [
#     [1,1], [1,3], [1,2], [6], [8], [8], [1,4,1], [2], [4], [6]
# ]






































