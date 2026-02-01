import numpy as np

def calculate_calibration_metrics(confidence_scores, accuracies, n_bins=5):
    """
    Computes Expected Calibration Error (ECE) and binning data.
    
    Args:
        confidence_scores (list): List of confidence scores (0-1).
        accuracies (list): List of binary accuracy values (0 or 1).
        n_bins (int): Number of bins.
        
    Returns:
        dict: ECE score and bin data for plotting.
    """
    if not confidence_scores or not accuracies:
        return {"ece": 0.0, "bins": []}
        
    confidences = np.array(confidence_scores)
    accs = np.array(accuracies)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    bin_data = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        # Indices of samples in this bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accs[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_data.append({
                "avg_confidence": float(avg_confidence_in_bin),
                "accuracy": float(accuracy_in_bin),
                "count": int(np.sum(in_bin))
            })
        else:
            bin_data.append({
                "avg_confidence": 0.0,
                "accuracy": 0.0,
                "count": 0
            })
            
    return {"ece": float(ece), "bins": bin_data}
