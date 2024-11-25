import cv2
import numpy as np
import json

def save_calibration_to_json(file_path, camera_matrix, dist_coeffs, rvecs, tvecs):
    """
    Save the camera calibration results to a JSON file.

    Args:
    - file_path: Path to the JSON file.
    - camera_matrix: Intrinsic parameters of the camera.
    - dist_coeffs: Distortion coefficients.
    - rvecs: Rotation vectors (extrinsic parameters for each image).
    - tvecs: Translation vectors (extrinsic parameters for each image).
    """
    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "rotation_vectors": [rvec.tolist() for rvec in rvecs],
        "translation_vectors": [tvec.tolist() for tvec in tvecs]
    }
    with open(file_path, "w") as f:
        json.dump(calibration_data, f, indent=4)
    print(f"Calibration data saved to {file_path}")


def live_camera_calibration(camera_id, grid_size, square_size, max_samples=20):
    """
    Perform live calibration of a camera using a video feed.

    Args:
    - camera_id: ID of the camera to calibrate (e.g., 0 or 1).
    - grid_size: Size of the chessboard grid (columns, rows of internal corners).
    - square_size: Size of each chessboard square (in mm).
    - max_samples: Number of samples to collect for calibration.

    Returns:
    - camera_matrix: Intrinsic parameters of the camera.
    - dist_coeffs: Distortion coefficients.
    - rvecs: Rotation vectors for each calibration image.
    - tvecs: Translation vectors for each calibration image.
    """
    # Prepare object points
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

    # Lists to store object points and image points
    objpoints = []
    imgpoints = []

    # Open the camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return None, None, None, None

    print("Starting live calibration... Move the chessboard in front of the camera.")
    print(f"Collecting up to {max_samples} valid frames.")

    collected_samples = 0
    while collected_samples < max_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if ret:
            # Refine corner positions
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Draw corners on the frame
            cv2.drawChessboardCorners(frame, grid_size, corners_refined, ret)

            # If 'space' is pressed, store the sample
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                collected_samples += 1
                print(f"Sample {collected_samples}/{max_samples} collected.")

        # Display the video feed
        cv2.imshow('Live Calibration', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Calibration canceled by user.")
            cap.release()
            cv2.destroyAllWindows()
            return None, None, None, None

    cap.release()
    cv2.destroyAllWindows()

    # Perform camera calibration
    print("Performing camera calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret:
        print("Calibration successful!")
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coeffs)
        return camera_matrix, dist_coeffs, rvecs, tvecs
    else:
        print("Calibration failed.")
        return None, None, None, None


# Example usage
if __name__ == "__main__":
    grid_size = (7, 9)  # Internal corners (columns, rows)
    square_size = 25  # Chessboard square size in mm
    camera_id = 0  # ID of the camera to calibrate
    output_file = "camera_calibration.json"  # Output JSON file for saving calibration data

    camera_matrix, dist_coeffs, rvecs, tvecs = live_camera_calibration(camera_id, grid_size, square_size, max_samples=20)

    if camera_matrix is not None:
        print("Calibration completed successfully.")
        save_calibration_to_json(output_file, camera_matrix, dist_coeffs, rvecs, tvecs)
