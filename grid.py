import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyautogui
from PIL import Image
import numpy as np
import logging
import cv2


class Cell:
    """
    Represents a single cell in the grid.
    """
    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color  # RGB tuple
        self.neighbors = {}
        self.group = None  # Reference to the group this cell belongs to

    @property
    def label(self):
        column = chr(ord('A') + self.col)
        row = self.row + 1
        return f"{column}{row}"

    def __repr__(self):
        return f"Cell(position={self.label}, color={self.color}, group={self.group})"


class GridGroup:
    """
    Represents a group of connected cells with the same color.
    """
    def __init__(self, color):
        self.color = color  # RGB tuple
        self.cells = []

    def add_cell(self, cell):
        """
        Adds a cell to the group.
        """
        self.cells.append(cell)
        cell.group = self.color  # Assign the group identifier to the cell

    def __repr__(self):
        return f"GridGroup(color={self.color}, size={len(self.cells)})"


class Grid:
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
            return sum((c1[i] - c2[i]) ** 2 for i in range(3))

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

    def visualize(self, title="Grid Visualization"):
        """
        Visualizes the grid and groups using matplotlib.
        Each cell is colored based on its RGB value, and grid labels are displayed.
        """
        fig, ax = plt.subplots(figsize=(self.cols, self.rows))
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
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

                    # Add grid location label (e.g., "A1", "B2")
                    ax.text(
                        col + 0.5,
                        flipped_row + 0.5,
                        cell.label,
                        color="black",
                        ha="center",
                        va="center",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"),
                    )

        # Add title and show the plot
        ax.set_title(title)
        plt.show()


def snap_to_palette(color, palette):
    """
    Snaps an RGB color to the nearest color in the given palette.
    :param color: A tuple (R, G, B) representing the color.
    :param palette: A list of RGB tuples representing the color palette.
    :return: The nearest color in the palette.
    """
    def color_distance(c1, c2):
        return sum((c1[i] - c2[i]) ** 2 for i in range(3))

    return min(palette, key=lambda p: color_distance(color, p))


def quantize_colors_with_palette(cells, palette):
    """
    Reduces the number of unique colors in the grid by snapping to a predefined palette.
    :param cells: A 2D list of RGB tuples representing the grid colors.
    :param palette: A list of RGB tuples representing the color palette.
    :return: A 2D list of quantized RGB tuples.
    """
    quantized_cells = []
    for row in cells:
        quantized_row = [snap_to_palette(color, palette) for color in row]
        quantized_cells.append(quantized_row)
    return quantized_cells