import numpy as np

def fit_circle(points):
    """Fits a circle to a set of 2D points using algebraic estimation."""
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # Solve the linear system: x^2 + y^2 + Ax + By + C = 0
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x**2 + y**2)
    
    # Solve for [A, B, C] using least squares
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = -0.5 * coeffs[:2]  # Compute the center
    r = np.sqrt(np.mean((x - xc) ** 2 + (y - yc) ** 2))  # Compute the radius
    
    return (xc, yc), r, np.sqrt((x - xc)**2 + (y - yc)**2)

def circularity_score(points):
    """Evaluates how circularly distributed the given points are."""
    if len(points) < 3:
        return 0  # At least 3 points are needed to form a circle

    center, radius, distances = fit_circle(points)

    # 1. Radial consistency: Standard deviation of distances from the center
    radial_std = np.std(distances) / radius  # Normalize by radius

    # 2. Angular consistency: Check if points are evenly spaced
    angles = np.arctan2([p[1] - center[1] for p in points], 
                         [p[0] - center[0] for p in points])
    angles = np.sort(angles)  # Sort to check spacing
    angle_diffs = np.diff(np.append(angles, angles[0] + 2 * np.pi))  # Circular difference

    ideal_spacing = 2 * np.pi / len(points)
    angular_std = np.std(angle_diffs) / ideal_spacing  # Normalize by ideal spacing

    # Combine both criteria (1 means perfect circle)
    score = 1 - (radial_std + angular_std) / 2
    return max(0, min(1, score))  # Ensure score is between 0 and 1

points_circle = [(np.cos(θ), np.sin(θ)) for θ in np.linspace(0, 2*np.pi, 10, endpoint=False)]
points_random1 = np.random.rand(4, 2)  # Randomly scattered points
points_random2 = np.random.rand(8, 2)  # Randomly scattered points
points_random3 = np.random.rand(16, 2)  # Randomly scattered points

print(circularity_score(points_circle))  # Should be close to 1
print(circularity_score(points_random1))  # Should be close to 0
print(circularity_score(points_random2))  # Should be close to 0
print(circularity_score(points_random3))  # Should be close to 0
