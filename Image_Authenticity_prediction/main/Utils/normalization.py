import numpy as np

def normalize_data(data: np.ndarray, min_range: float, max_range: float) -> np.ndarray:
    """
    Normalizes a numpy array to a specified range, either [0, 1] or [-1, 1].
    """
    if not ((min_range == 0 and max_range == 1) or (min_range == -1 and max_range == 1)):
        raise ValueError("Normalization is only supported for ranges [0, 1] and [-1, 1].")

    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max > data_min:
        # Scale data to [0, 1] first
        scaled_0_1 = (data - data_min) / (data_max - data_min)
        # Then scale to [min_range, max_range]
        return scaled_0_1 * (max_range - min_range) + min_range
    else:
        # If all elements are the same, return an array of the midpoint of the range
        return np.full_like(data, (min_range + max_range) / 2)


