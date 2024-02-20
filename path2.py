import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

def create_closed_loop_path(num_curves, box_width, box_height):
    # Ensure num_curves is even to end up where we started
    if num_curves % 2 == 1:
        num_curves += 1
    
    # Generate random control points within the box
    control_points = np.random.rand(num_curves + 1, 2)
    control_points[:, 0] *= box_width
    control_points[:, 1] *= box_height
    
    # Make the last control point the same as the first to close the loop
    control_points[-1] = control_points[0]

    # Define path commands
    codes = [Path.MOVETO] + [Path.CURVE4] * 3 * num_curves + [Path.CLOSEPOLY]

    # Flatten the control points list
    vertices = control_points.flatten().reshape(-1, 2)
    
    # Add the first point two more times, as it's needed for the cubic Bézier curve
    vertices = np.vstack([vertices, [vertices[0], vertices[0]]])
    
    # Create the path
    path = Path(vertices, codes)

    return path

def plot_path(path):
    fig, ax = plt.subplots()
    patch = patches.PathPatch(path, facecolor='none', edgecolor='black')
    ax.add_patch(patch)
    ax.set_xlim(0, box_width)
    ax.set_ylim(0, box_height)
    ax.set_aspect('equal')
    plt.show()

# Define the number of curves and the box dimensions
num_curves = 6
box_width = 10
box_height = 10

# Create and plot the closed loop path
closed_loop_path = create_closed_loop_path(num_curves, box_width, box_height)
plot_path(closed_loop_path)
