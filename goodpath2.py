import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

def bezier_point(t, control_points):
    """Calculate the point on a Bezier curve for a given value of t."""
    n = len(control_points) - 1
    point = np.zeros(2)
    for i, p in enumerate(control_points):
        binomial = binom(n, i)
        factor = binomial * (t ** i) * ((1 - t) ** (n - i))
        point += factor * p
    return point

def bezier_curve(control_points, num_samples=100):
    """Sample points on a Bezier curve defined by control points."""
    return np.array([bezier_point(t, control_points) for t in np.linspace(0, 1, num_samples)])

def bezier_normal(t, control_points):
    """Calculate the normal to a Bezier curve at a given t."""
    n = len(control_points) - 1
    derivative = np.zeros(2)
    for i in range(n):
        binomial = binom(n - 1, i)
        factor = binomial * (t ** i) * ((1 - t) ** (n - 1 - i))
        derivative += factor * (control_points[i + 1] - control_points[i]) * n
    tangent = np.array([derivative[1], -derivative[0]])
    normal = tangent / np.linalg.norm(tangent)
    return normal

def offset_curve(control_points, offset_distance, num_samples=100):
    """Create an offset Bezier curve by a fixed distance."""
    original_curve = bezier_curve(control_points, num_samples)
    offset_points = np.array([bezier_point(t, control_points) + offset_distance * bezier_normal(t, control_points) for t in np.linspace(0, 1, num_samples)])
    return original_curve, offset_points

def adjust_offset_curve(original_curve, offset_curve, threshold):
    """Adjust the offset curve points if they are too close to the original curve."""
    adjusted_curve = np.copy(offset_curve)
    for i in range(len(offset_curve)):
        distance_vector = offset_curve[i] - original_curve[i]
        distance = np.linalg.norm(distance_vector)
        if distance < threshold:
            adjusted_distance = threshold - distance
            adjusted_curve[i] += adjusted_distance * (distance_vector / distance)
    return adjusted_curve

# Add the plotting code here using the provided functions
# Example usage:
control_points = np.array([
    [0.2, 0.2],
    [0.2, 0.8],
    [0.8, 0.8],
    [0.8, 0.2]
])

num_samples = 100
offset_distance = 0.05  # Distance to offset from the original curve
threshold = 0.02  # Minimum distance threshold

# Generate the original and offset curves
original_curve, initial_offset_curve = offset_curve(control_points, offset_distance, num_samples)
adjusted_offset_curve = adjust_offset_curve(original_curve, initial_offset_curve, threshold)

# Plot the curves
plt.plot(original_curve[:, 0], original_curve[:, 1], label='Original Curve')
plt.plot(initial_offset_curve[:, 0], initial_offset_curve[:, 1], label='Initial Offset Curve')
plt.plot(adjusted_offset_curve[:, 0], adjusted_offset_curve[:, 1], label='Adjusted Offset Curve')

plt.legend()
plt.show()
