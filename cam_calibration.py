import cv2
import numpy as np
import glob
import os

def calibrate_camera(images, grid_size, square_size):
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    for image_path in images:
        print(f"Processing {image_path}")
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
        print(f"Corners detected: {ret}")

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(image, grid_size, corners, ret)
            cv2.imshow('Chessboard', image)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("No valid points for calibration. Check your images and grid_size.")
        return None, None, None, None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("Calibration successful!")
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coeffs)
    else:
        print("Calibration failed.")

    return camera_matrix, dist_coeffs, rvecs, tvecs


# **6. Charger toutes les images depuis le dossier `chessboard images`**
image_folder = "chessboard images"  # Nom du dossier contenant les images
image_extensions = ["*.jpg"]  # Extensions d'image possibles
image_paths = []

# Rechercher toutes les images dans le dossier
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(image_folder, ext)))

# Trier les chemins pour un ordre cohérent
image_paths.sort()

# Taille de la grille de l'échiquier (colonnes, lignes de coins intérieurs)
grid_size = (7, 9)

# Taille réelle des cases de l'échiquier (en millimètres)
square_size = 20  # 25 mm

# Lancer la calibration
camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(image_paths, grid_size, square_size)

