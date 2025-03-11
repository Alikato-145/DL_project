import numpy as np
from sklearn.metrics import mean_absolute_error

def calculate_mae(real_count, predicted_count):
    """
    Calculate Mean Absolute Error (MAE) between real and predicted counts.

    Args:
    - real_count: Dictionary of real counts
    - predicted_count: Dictionary of predicted counts

    Returns:
    - MAE value and detailed metrics
    """
    # Create a set of all unique colors
    all_colors = set(real_count.keys()).union(set(predicted_count.keys()))
    
    # Calculate absolute errors for each color
    color_errors = {}
    total_error = 0
    
    for color in all_colors:
        real = real_count.get(color, 0)
        pred = predicted_count.get(color, 0)
        error = abs(real - pred)
        color_errors[color] = error
        total_error += error
    
    # Create arrays for real and predicted counts for sklearn MAE calculation
    real_counts = np.array([real_count.get(color, 0) for color in all_colors])
    predicted_counts = np.array([predicted_count.get(color, 0) for color in all_colors])
    
    # Calculate overall MAE
    mae = mean_absolute_error(real_counts, predicted_counts)
    
    # Calculate total counts
    total_real = sum(real_count.values())
    total_predicted = sum(predicted_count.values())
    
    return {
        "mae": float(mae),
        "color_errors": color_errors,
        "total_error": total_error,
        "total_real": total_real,
        "total_predicted": total_predicted,
        "absolute_total_error": abs(total_real - total_predicted)
    }
