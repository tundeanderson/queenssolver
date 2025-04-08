import logging
import pyautogui
import numpy as np
import cv2
from PIL import Image
from grid import Grid
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def detect_grid_size(image):
    """
    Detects the grid size (n x n) by analyzing the image for grid lines.
    """
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, threshold1=350, threshold2=400)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=475, maxLineGap=50)

    if lines is None:
        raise ValueError("No grid lines detected.")

    horizontal_lines = set()
    vertical_lines = set()

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10:  # Horizontal line
            horizontal_lines.add(y1)
        elif abs(x1 - x2) < 10:  # Vertical line
            vertical_lines.add(x1)

    grid_size = min(len(horizontal_lines) - 1, len(vertical_lines) - 1)
    logging.info(f"Detected grid size: {grid_size} x {grid_size}")
    logging.debug(f"Horizontal lines: {sorted(horizontal_lines)}")
    logging.debug(f"Vertical lines: {sorted(vertical_lines)}")
    return grid_size


def capture_and_analyze_grid(region):
    """
    Captures a section of the screen, detects the grid size, and identifies colors.
    """
    screenshot = pyautogui.screenshot(region=region)
    image = Image.fromarray(np.array(screenshot))
    grid_size = detect_grid_size(image)

    cell_width = region[2] // grid_size
    cell_height = region[3] // grid_size

    cells = []
    for row in range(grid_size):
        row_colors = []
        for col in range(grid_size):
            # Calculate the center of the cell
            center_x = col * cell_width + cell_width // 2
            center_y = row * cell_height + cell_height // 2

            # Get the exact color at the center of the cell
            pixel_color = image.getpixel((center_x, center_y))
            row_colors.append(pixel_color)  # Use the exact color
        cells.append(row_colors)

    logging.info(f"Grid colors captured for a {grid_size} x {grid_size} grid.")
    logging.debug(f"Extracted cells: {cells}")
    return cells


def solve(grid):
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
        if group_index == len(grid.groups):
            return True  # All groups processed successfully

        group = grid.groups[group_index]
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
    for group in grid.groups:
        group.identified_cell = None  # Reset identified cells

    if not backtrack(0):
        raise ValueError("No solution exists for the given grid.")

    logging.info("Solution found!")
    for group in grid.groups:
        logging.info(f"Group {group.color}: Identified Cell = ({group.identified_cell.row}, {group.identified_cell.col})")


if __name__ == "__main__":
    region = (389, 265, 502, 501)  # Example region
    try:
        logging.info("Starting grid detection...")

        cells = capture_and_analyze_grid(region)
        grid = Grid(cells)

        logging.info("Grid successfully built.")
        grid.log_groups()  # Log group details

        solve(grid)  # Solve the Queens game
        grid.visualize(title="Grid and Groups Visualization")

    except ValueError as e:
        logging.error(f"Error: {e}")