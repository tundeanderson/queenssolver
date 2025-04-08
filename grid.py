import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

class Grid():
    """
    Represents the entire grid of cells.
    """
    def __init__(self, cells):
        self.rows = len(cells)
        self.cols = len(cells[0]) if self.rows > 0 else 0
        self.cells = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.groups = []  # List of GridGroup objects
        self.build(cells)

    def set_cell(self, row, col, cell):
        self.cells[row][col] = cell

    def get_cell(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.cells[row][col]
        return None

    def get_cell_by_label(self, label):
        """
        Retrieves a cell object by its label.
        """
        for row in self.cells:
            for cell in row:
                if cell.label == label:
                    return cell
        return None

    def build(self, cells):
        """
        Builds the grid by creating cells and assigning them to groups.
        """
        # Create cells
        for row in range(self.rows):
            for col in range(self.cols):
                color = cells[row][col]
                cell = Cell(row, col, color)
                self.set_cell(row, col, cell)

        logging.debug(f"Grid structure after build: {self.cells}")

        # Link neighbors
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.get_cell(row, col)
                if row > 0:
                    cell.neighbors["up"] = self.get_cell(row - 1, col)
                if row < self.rows - 1:
                    cell.neighbors["down"] = self.get_cell(row + 1, col)
                if col > 0:
                    cell.neighbors["left"] = self.get_cell(row, col - 1)
                if col < self.cols - 1:
                    cell.neighbors["right"] = self.get_cell(row, col + 1)

        # Group cells
        self.group_cells()

    def group_cells(self, color_similarity_threshold=1000):
        """
        Groups cells into GridGroup objects based on color similarity and adjacency.
        :param color_similarity_threshold: The maximum allowed color distance for cells to be in the same group.
        """
        visited = set()

        def color_distance(c1, c2):
            """
            Calculates the squared Euclidean distance between two RGB colors.
            """
            return sum((int(c1[i]) - int(c2[i])) ** 2 for i in range(3))

        def dfs(cell, group):
            """
            Depth-first search to find all connected cells of similar color.
            """
            stack = [cell]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                group.add_cell(current)
                for neighbor in current.neighbors.values():
                    if neighbor and neighbor not in visited:
                        if color_distance(current.color, neighbor.color) <= color_similarity_threshold:
                            stack.append(neighbor)

        # Iterate through all cells and group them
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.get_cell(row, col)
                if cell not in visited:
                    group = GridGroup(cell.color)
                    dfs(cell, group)
                    self.groups.append(group)

    
    def solve(self):
        """
        Solves the Queens game by identifying one cell in each group such that:
        - Exactly one cell is identified in each row, column, and group.
        - No two identified cells are adjacent (including diagonally).
        Populates each GridGroup with the identified cell.
        """
        identified_cells = set()  # Set of identified Cell objects
        rows_used = set()  # Rows that already have an identified cell
        cols_used = set()  # Columns that already have an identified cell

        def is_valid(cell):
            """
            Checks if a cell can be identified without violating the rules.
            """
            # Check if the cell's row or column is already used
            if cell.row in rows_used or cell.col in cols_used:
                return False

            # Check adjacency (including diagonals)
            for identified_cell in identified_cells:
                if abs(identified_cell.row - cell.row) <= 1 and abs(identified_cell.col - cell.col) <= 1:
                    return False

            return True

        def backtrack(group_index):
            """
            Backtracking function to identify one cell per group.
            """
            if group_index == len(self.groups):
                return True  # All groups processed successfully

            group = self.groups[group_index]
            for cell in group.cells:
                if is_valid(cell):
                    # Mark the cell as identified
                    identified_cells.add(cell)
                    rows_used.add(cell.row)
                    cols_used.add(cell.col)
                    group.identified_cell = cell

                    # Recurse to the next group
                    if backtrack(group_index + 1):
                        return True

                    # Backtrack: unmark the cell
                    identified_cells.remove(cell)
                    rows_used.remove(cell.row)
                    cols_used.remove(cell.col)
                    group.identified_cell = None

            return False  # No valid cell found for this group

        # Initialize the backtracking process
        for group in self.groups:
            group.identified_cell = None  # Reset identified cells

        if not backtrack(0):
            raise ValueError("No solution exists for the given grid.")

        logging.info("Solution found!")
        for group in self.groups:
            logging.info(f"Group {group.color}: Identified Cell = ({group.identified_cell.row}, {group.identified_cell.col})")

    def visualize(self, title="Grid Visualization"):
        """
        Visualizes the grid and groups using matplotlib.
        Each cell is colored based on its RGB value, and axis labels are displayed.
        Identified cells are marked with a queen's crown icon.
        """
        fig, ax = plt.subplots(figsize=(self.cols, self.rows))
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_xticks(range(self.cols + 1))  # Include boundary ticks
        ax.set_yticks(range(self.rows + 1))  # Include boundary ticks
        ax.set_xticklabels([''] + [chr(ord('A') + i) for i in range(self.cols)], fontsize=10)  # Add empty label for boundary
        ax.set_yticklabels([''] + [str(i + 1) for i in range(self.rows)], fontsize=10)  # Add empty label for boundary
        ax.grid(color="black", linestyle="-", linewidth=1)

        # Draw each cell
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.get_cell(row, col)
                if cell:
                    # Flip the row index to invert the grid on the horizontal axis
                    flipped_row = self.rows - row - 1

                    # Convert RGB to normalized color (0-1 range for matplotlib)
                    normalized_color = tuple(c / 255 for c in cell.color)
                    rect = mpatches.Rectangle((col, flipped_row), 1, 1, color=normalized_color)
                    ax.add_patch(rect)

                    # Add a marker for identified cells
                    for group in self.groups:
                        if group.identified_cell == cell:
                            ax.text(
                                col + 0.5,
                                flipped_row + 0.5,  # Centered in the cell
                                "♛",  # Queen's crown icon
                                color="black",
                                ha="center",
                                va="center",
                                fontsize=24,
                                fontweight="bold",
                            )

        # Add title and show the plot
        ax.set_title(title)
        plt.show()

    def log_groups(self):
        """
        Logs the number of groups, the number of cells in each group, their colors, and the cell positions in each group.
        """
        logging.info(f"Total number of groups: {len(self.groups)}")
        for i, group in enumerate(self.groups, start=1):
            cell_positions = [(cell.row, cell.col) for cell in group.cells]
            logging.info(f"Group {i}: Color={group.color}, Size={len(group.cells)}, Cells={cell_positions}")

class Cell():
    """
    Represents a single cell in the grid.
    """
    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color  # RGB tuple
        self.neighbors = {}
        self.group = None  # Reference to the group this cell belongs to

    def __repr__(self):
        return f"Cell(row={self.row}, col={self.col}, group={self.group})"

class GridGroup():
    """
    Represents a group of connected cells with the same color.
    """
    def __init__(self, color):
        self.color = color  # RGB tuple
        self.cells = []
        self.identified_cell = None  # Optional marker for a specific cell

    def add_cell(self, cell):
        """
        Adds a cell to the group.
        """
        self.cells.append(cell)
        cell.group = self.color  # Assign the group identifier to the cell

    def __repr__(self):
        return f"GridGroup(color={self.color}, size={len(self.cells)})"
