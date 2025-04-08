import logging
import pyautogui
import numpy as np
import cv2
from PIL import Image
from grid import Grid
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def detect_grid_region():
    """
    Detects the region of the grid on the screen, assuming it is encased within a black border.
    :return: A tuple (x, y, width, height) representing the grid's region.
    """
    # Capture the entire screen
    screenshot = pyautogui.screenshot()
    screen_image = np.array(screenshot)

    # Convert the screenshot to grayscale
    gray_screen = cv2.cvtColor(screen_image, cv2.COLOR_RGB2GRAY)

    # Apply binary thresholding to highlight the black border
    _, binary_image = cv2.threshold(gray_screen, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangular contour (assumed to be the grid)
    grid_x, grid_y, grid_width, grid_height = 0, 0, 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Check if the contour is large enough to be the grid
        if w > grid_width and h > grid_height and w == h:  # Adjust size thresholds as needed
            grid_x, grid_y, grid_width, grid_height = x, y, w, h

    if grid_width == 0 or grid_height == 0:
        raise ValueError("Grid not found on the screen.")

    # Debug: Draw the detected grid region on the original image and save it
    debug_image = screen_image.copy()
    cv2.rectangle(debug_image, (grid_x, grid_y), (grid_x + grid_width, grid_y + grid_height), (0, 255, 0), 2)
    debug_image_path = "grid_region_debug.png"
    cv2.imwrite(debug_image_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    logging.info(f"Debug image with detected grid region saved to {debug_image_path}")

    logging.info(f"Detected grid region: (x={grid_x}, y={grid_y}, width={grid_width}, height={grid_height})")
    return grid_x, grid_y, grid_width, grid_height


def capture_and_analyze_grid(grid_region):
    """
    Captures a section of the screen, detects the grid size, and identifies grid lines.
    :param grid_region: A tuple (x, y, width, height) representing the grid's region.
    :return: A 2D list of cell colors representing the grid.
    """
    x, y, width, height = grid_region

    # Capture the grid region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    image = np.array(screenshot)

    # Save the cropped region for debugging
    debug_image_path = "cropped_grid_region.png"
    cv2.imwrite(debug_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logging.info(f"Cropped grid region saved to {debug_image_path}")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply binary thresholding to highlight grid lines
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)

    # Save the binary image for debugging
    cv2.imwrite("binary_grid_debug.png", binary_image)

    # Sum pixel intensities along rows and columns
    row_sums = np.sum(binary_image, axis=1)  # Sum along rows
    col_sums = np.sum(binary_image, axis=0)  # Sum along columns

    # Identify peaks in the intensity profiles
    def find_peaks(sums, axis_length):
        """
        Finds grid lines by analyzing the intensity profile.
        :param sums: The summed pixel intensities along rows or columns.
        :param axis_length: The length of the axis (height for rows, width for columns).
        :return: A list of detected line positions.
        """
        threshold = np.max(sums) * 0.5  # Threshold for significant intensity
        lines = [i for i, value in enumerate(sums) if value > threshold]

        # Group nearby lines to account for line thickness
        grouped_lines = []
        current_group = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i] - lines[i - 1] <= axis_length // 50:  # Adjust grouping threshold dynamically
                current_group.append(lines[i])
            else:
                grouped_lines.append(int(np.mean(current_group)))  # Use the average position of the group
                current_group = [lines[i]]
        grouped_lines.append(int(np.mean(current_group)))  # Add the last group
        return grouped_lines

    horizontal_lines = find_peaks(row_sums, height)
    vertical_lines = find_peaks(col_sums, width)

    # Ensure the number of horizontal and vertical lines is equal
    if len(horizontal_lines) != len(vertical_lines):
        raise ValueError("Mismatch between detected horizontal and vertical lines.")

    # Calculate the grid size
    grid_size = len(horizontal_lines) - 1  # Number of cells is one less than the number of lines
    logging.info(f"Detected grid size: {grid_size} x {grid_size}")

    # Extract cell colors
    cell_width = width // grid_size
    cell_height = height // grid_size
    cells = []
    for row in range(grid_size):
        cell_row = []
        for col in range(grid_size):
            # Calculate the center of the cell
            center_x = col * cell_width + cell_width // 2
            center_y = row * cell_height + cell_height // 2

            # Get the color at the center of the cell
            pixel_color = image[center_y, center_x]
            cell_row.append(tuple(pixel_color))  # Convert to RGB tuple
        cells.append(cell_row)

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
    try:
        logging.info("Starting grid detection...")

        # Detect the grid region dynamically
        grid_region = detect_grid_region()
        
        # Capture and analyze the grid
        cells = capture_and_analyze_grid(grid_region)
        print(cells)
        grid = Grid(cells)

        # logging.info("Grid successfully built.")
        grid.log_groups()  # Log group details

        solve(grid)  # Solve the Queens game
        grid.visualize(title="Grid and Groups Visualization")

    except ValueError as e:
        logging.error(f"Error: {e}")