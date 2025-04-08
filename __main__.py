import logging
import pyautogui
import numpy as np
import cv2
import time
from grid import Grid

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

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply binary thresholding to highlight grid lines
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)

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
    
    # LinkedIn preview workaround
    if len(horizontal_lines) < 5 or len(vertical_lines) < 5:
        raise ValueError("Too few grid lines detected")

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

if __name__ == "__main__":
    try:
        logging.info("Starting grid detection...")

        # Wait for puzzle to appear on screen
        while True:
            try:
                # Detect the grid region dynamically
                grid_region = detect_grid_region()
                
                # Capture and analyze the grid
                cells = capture_and_analyze_grid(grid_region)
                break
            except:
                logging.info("No grid detected, waiting...")
                time.sleep(1)
        
        grid = Grid(cells, grid_region)
        grid.solve()

    except ValueError as e:
        logging.error(f"Error: {e}")