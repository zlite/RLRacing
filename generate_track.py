import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Function to add a straight segment to the track
def add_straight_segment(ax, start_point, length, angle):
    end_point = start_point + np.array([length * np.cos(angle), length * np.sin(angle)])
    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-')
    return end_point

# Function to add a curved segment to the track
def add_curved_segment(ax, start_point, radius, angle, turn_direction):
    if turn_direction == 'left':
        center = start_point + np.array([radius * np.sin(angle), -radius * np.cos(angle)])
        theta1, theta2 = np.degrees(angle), np.degrees(angle + np.pi/2)
        angle += np.pi/2
    else:
        center = start_point + np.array([-radius * np.sin(angle), radius * np.cos(angle)])
        theta1, theta2 = np.degrees(angle), np.degrees(angle - np.pi/2)
        angle -= np.pi/2

    arc = patches.Arc(center, 2*radius, 2*radius, angle=0, theta1=theta1, theta2=theta2, color='k')
    ax.add_patch(arc)
    end_point = np.array([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)])
    return end_point, angle

# Function to generate a random track that forms a closed loop
def generate_random_closed_loop_track(ax, num_tiles):
    start_point = np.array([0, 0])
    angle = 0
    lengths = np.random.uniform(2, 5, num_tiles)
    radii = np.random.uniform(1, 3, num_tiles)
    directions = np.random.choice(['left', 'right'], num_tiles)
    
    # Ensure we have an equal number of left and right turns
    left_turns = np.count_nonzero(directions == 'left')
    right_turns = num_tiles - left_turns
    while left_turns != right_turns:
        # Adjust the directions to balance the turns
        if left_turns < right_turns:
            directions[np.random.choice(np.where(directions == 'right')[0])] = 'left'
        else:
            directions[np.random.choice(np.where(directions == 'left')[0])] = 'right'
        left_turns = np.count_nonzero(directions == 'left')
        right_turns = num_tiles - left_turns

    for i in range(num_tiles):
        start_point = add_straight_segment(ax, start_point, lengths[i], angle)
        start_point, angle = add_curved_segment(ax, start_point, radii[i], angle, directions[i])

    # Close the loop
    end_point, angle = add_curved_segment(ax, start_point, radii[-1], angle, 'right' if directions[-1] == 'left' else 'left')
    add_straight_segment(ax, end_point, np.linalg.norm(start_point), angle)

# Create plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.axis('off')

# Generate and plot random closed loop track
generate_random_closed_loop_track(ax, 8)

plt.show()
