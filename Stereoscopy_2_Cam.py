import cv2
import numpy as np
import glob
import os

# **1. Fonction de calibration de la caméra**
def calibrate_camera(image_folder, grid_size, square_size):
    """
    Calibre une caméra à l'aide d'images d'un échiquier.

    Args:
    - image_folder : Dossier contenant les images d'échiquier.
    - grid_size : Taille de la grille (colonnes, lignes de coins intérieurs).
    - square_size : Taille d'une case de l'échiquier (en millimètres).

    Returns:
    - camera_matrix : Matrice intrinsèque de la caméra.
    - dist_coeffs : Coefficients de distorsion.
    """
    # Initialisation des points 3D de l'échiquier (z=0 car il est plat)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # Liste des points 3D réels
    imgpoints = []  # Liste des points 2D détectés dans les images

    # Charger les images du dossier
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    if not image_paths:
        print(f"Aucune image trouvée dans le dossier {image_folder}.")
        return None, None

    for image_path in image_paths:
        print(f"Processing {image_path}...")
        image = cv2.imread(image_path)  # Charger l'image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris

        # Trouver les coins de l'échiquier
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if ret:
            objpoints.append(objp)  # Ajouter les points 3D
            imgpoints.append(corners)  # Ajouter les points 2D

            # Dessiner les coins détectés pour vérification visuelle
            cv2.drawChessboardCorners(image, grid_size, corners, ret)
            cv2.imshow('Chessboard', image)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("Échec de la détection des coins d'échiquier.")
        return None, None

    # Calibration de la caméra
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret:
        print("Calibration réussie !")
        print("Matrice intrinsèque :\n", camera_matrix)
        print("Coefficients de distorsion :\n", dist_coeffs)
        return camera_matrix, dist_coeffs
    else:
        print("Échec de la calibration.")
        return None, None


# **2. Calcul de la position 3D d'un objet**
def calculate_3d_position(uL, uR, vL, focal_length, baseline, cx, cy):
    """
    Calcule la position 3D (X, Y, Z) d'un point en utilisant la stéréovision.

    Args:
    - uL, uR : Coordonnées X du point dans les images de gauche et de droite.
    - vL : Coordonnée Y dans l'image de gauche.
    - focal_length : Distance focale de la caméra.
    - baseline : Distance entre les deux caméras (en millimètres).
    - cx, cy : Coordonnées du centre optique.

    Returns:
    - X, Y, Z : Coordonnées du point dans l'espace 3D.
    """
    disparity = uL - uR  # Différence entre les positions du point dans les deux images
    if disparity == 0:
        raise ValueError("La disparité ne peut pas être nulle.")  # Empêche la division par zéro
    Z = (baseline * focal_length) / disparity  # Profondeur (Z)
    X = Z * (uL - cx) / focal_length  # Position X
    Y = Z * (vL - cy) / focal_length  # Position Y
    return X, Y, Z


# **3. Détection des objets rouges dans une image**
def detect_red_objects(frame, lower_red1, upper_red1, lower_red2, upper_red2):
    """
    Détecte des objets rouges dans une image et retourne leurs centres.

    Args:
    - frame : Image à analyser.
    - lower_red1, upper_red1 : Plage HSV pour le rouge vif.
    - lower_red2, upper_red2 : Plage HSV pour le rouge sombre.

    Returns:
    - frame : Image annotée avec des rectangles et points.
    - centers : Liste des centres des objets rouges détectés.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convertir en espace de couleur HSV
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # Masque pour rouge vif
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # Masque pour rouge sombre
    mask = cv2.bitwise_or(mask1, mask2)  # Combiner les deux masques

    # Nettoyer le masque pour éliminer le bruit
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Trouver les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    # Parcourir les contours détectés
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filtrer les objets trop petits
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # Centre en X
                cy = int(M["m01"] / M["m00"])  # Centre en Y
                centers.append((cx, cy))  # Ajouter le centre à la liste
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Dessiner un point bleu

    return frame, centers


# **4. Script principal**
if __name__ == "__main__":
    # **Calibration de la caméra**
    image_folder = "chessboard images"  # Remplacez par le dossier contenant vos images
    grid_size = (7, 9)  # Taille de la grille d'échiquier
    square_size = 20  # Taille des cases en mm

    camera_matrix, dist_coeffs = calibrate_camera(image_folder, grid_size, square_size)

    if camera_matrix is not None:
        focal_length = camera_matrix[0, 0]  # Distance focale
        cx = camera_matrix[0, 2]  # Coordonnée X du centre optique
        cy = camera_matrix[1, 2]  # Coordonnée Y du centre optique
        baseline = 240  # Distance entre les caméras en mm
    else:
        print("Échec de la calibration de la caméra.")
        exit()

    # **Plages de couleur rouge (HSV)**
    lower_red1 = (0, 150, 100)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 150, 100)
    upper_red2 = (180, 255, 255)

    # **Capture vidéo des deux caméras**
# **Capture vidéo des deux caméras**
cap_left = cv2.VideoCapture(0)  # Caméra gauche
cap_right = cv2.VideoCapture(1)  # Caméra droite

# Define fixed window names
cv2.namedWindow('Caméra Gauche', cv2.WINDOW_NORMAL)
cv2.namedWindow('Caméra Droite', cv2.WINDOW_NORMAL)

try:
    while cap_left.isOpened() and cap_right.isOpened():
        # Capture frames from both cameras
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        # Validate frames
        if not ret_left or frame_left is None:
            print("Invalid frame from Caméra Gauche. Exiting.")
            break
        if not ret_right or frame_right is None:
            print("Invalid frame from Caméra Droite. Exiting.")
            break

        # Detect red objects in the frames
        frame_left, centers_left = detect_red_objects(frame_left, lower_red1, upper_red1, lower_red2, upper_red2)
        frame_right, centers_right = detect_red_objects(frame_right, lower_red1, upper_red1, lower_red2, upper_red2)

        # Calculate 3D position if objects are detected in both cameras
        if centers_left and centers_right:
            uL, vL = centers_left[0]  # First detected object in Caméra Gauche
            uR, vR = centers_right[0]  # First detected object in Caméra Droite
            try:
                X, Y, Z = calculate_3d_position(uL, uR, vL, focal_length, baseline, cx, cy)
                print(f"Position 3D : X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")
            except ValueError as e:
                print(f"Error calculating 3D position: {e}")

        # Display annotated frames in fixed windows
        cv2.imshow('Caméra Gauche', frame_left)
        cv2.imshow('Caméra Droite', frame_right)

        # Limit frame rate and handle exit condition
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # Release resources and ensure all windows are closed
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    print("Resources released and all windows closed.")
