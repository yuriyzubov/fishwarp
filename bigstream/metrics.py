# Normalized Cross-Correlation

import numpy as np
from scipy.ndimage import correlate
from sklearn.metrics import mutual_info_score
import SimpleITK as sitk



def ncc(fixed, moving, mask=None):
    """
    Compute normalized cross-correlation between two images.
    
    Args:
        fixed: reference image
        moving: image to compare (after registration)
        mask: optional binary mask for region of interest
    
    Returns:
        NCC value (higher is better, max is 1.0)
    """
    if mask is not None:
        fixed = fixed[mask > 0]
        moving = moving[mask > 0]
    
    # Flatten arrays
    fixed = fixed.flatten()
    moving = moving.flatten()
    
    # Normalize by subtracting mean
    fixed_mean = np.mean(fixed)
    moving_mean = np.mean(moving)
    
    fixed_norm = fixed - fixed_mean
    moving_norm = moving - moving_mean
    
    # Compute correlation
    numerator = np.sum(fixed_norm * moving_norm)
    denominator = np.sqrt(np.sum(fixed_norm**2) * np.sum(moving_norm**2))
    
    return numerator / (denominator + 1e-10)

def mutual_information(fixed, moving, bins=32, mask=None):
    """
    Compute mutual information between two images.
    
    Args:
        fixed: reference image
        moving: image to compare
        bins: number of histogram bins
        mask: optional binary mask
    
    Returns:
        MI value (higher is better)
    """
    if mask is not None:
        fixed = fixed[mask > 0]
        moving = moving[mask > 0]
    
    fixed = fixed.flatten()
    moving = moving.flatten()
    
    # Discretize images into bins
    fixed_binned = np.digitize(fixed, bins=np.linspace(fixed.min(), fixed.max(), bins))
    moving_binned = np.digitize(moving, bins=np.linspace(moving.min(), moving.max(), bins))
    
    return mutual_info_score(fixed_binned, moving_binned)


def metrics_sitk(fixed, moving):
    """
    Compute metrics using SimpleITK.
    
    Args:
        fixed, moving: numpy arrays or SimpleITK images
    """
    # Convert to SimpleITK if needed
    if isinstance(fixed, np.ndarray):
        fixed = sitk.GetImageFromArray(fixed)
    if isinstance(moving, np.ndarray):
        moving = sitk.GetImageFromArray(moving)

    registration = sitk.ImageRegistrationMethod()

    # Correlation metric
    registration.SetMetricAsCorrelation()
    correlation = registration.MetricEvaluate(fixed, moving)
    
    # Mutual Information (Mattes)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    mi = registration.MetricEvaluate(fixed, moving)
    
    # Mean Squares
    registration.SetMetricAsMeanSquares()
    mse = registration.MetricEvaluate(fixed, moving)
    
    return {
        'correlation': correlation,
        'mutual_information': mi,
        'mean_squares': mse
    }



