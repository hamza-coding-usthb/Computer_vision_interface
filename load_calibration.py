import numpy as np
import json
def load_calibration_from_json(file_path):
    with open(file_path, "r") as f:
        calibration_data = json.load(f)
    camera_matrix = np.array(calibration_data["camera_matrix"])
    dist_coeffs = np.array(calibration_data["dist_coeffs"])
    rvecs = [np.array(rvec) for rvec in calibration_data["rotation_vectors"]]
    tvecs = [np.array(tvec) for tvec in calibration_data["translation_vectors"]]
    return camera_matrix, dist_coeffs, rvecs, tvecs

# Example usage
camera_matrix, dist_coeffs, rvecs, tvecs = load_calibration_from_json("camera_calibration.json")
