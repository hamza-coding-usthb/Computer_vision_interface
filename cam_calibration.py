import cv2
import numpy as np

def calibrate_camera(images, grid_size, square_size):
    """
    Calibre une caméra à l'aide d'images d'un échiquier.

    Args:
    - images : Liste des chemins vers les images d'échiquier.
    - grid_size : Taille de la grille de l'échiquier (nombre de coins intérieurs : (cols, rows)).
    - square_size : Taille d'une case de l'échiquier (en unités réelles, par ex. millimètres).

    Returns:
    - camera_matrix : Matrice intrinsèque de la caméra.
    - dist_coeffs : Coefficients de distorsion de l'objectif.
    - rvecs : Vecteurs de rotation pour chaque image.
    - tvecs : Vecteurs de translation pour chaque image.
    """
    # **1. Préparer les points 3D de l'échiquier**
    # - `objp` contient les coordonnées 3D des coins de l'échiquier dans le monde réel.
    # - Ici, on considère que l'échiquier est sur un plan plat à z = 0.
    # - La grille est définie en fonction de `grid_size` et de la taille réelle des cases (`square_size`).
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

    # **2. Initialiser les listes pour stocker les points 3D (objpoints) et 2D (imgpoints)**
    # - `objpoints` : Liste des positions 3D des coins dans le monde réel (inchangées pour toutes les images).
    # - `imgpoints` : Liste des positions 2D des coins détectés dans les images.
    objpoints = []  # Points 3D du monde
    imgpoints = []  # Points 2D dans les images

    # **3. Traiter chaque image d'échiquier fournie**
    for image_path in images:
        # **3.1 Charger l'image**
        image = cv2.imread(image_path)  # Charger l'image depuis le chemin
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris pour une meilleure précision

        # **3.2 Trouver les coins de l'échiquier**
        # - `cv2.findChessboardCorners` détecte les coins internes de l'échiquier.
        # - `grid_size` spécifie le nombre de coins intérieurs attendus dans la grille.
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        # **3.3 Si les coins sont détectés**
        if ret:
            # Ajouter les points 3D de l'échiquier (toujours les mêmes pour chaque image)
            objpoints.append(objp)

            # Ajouter les coordonnées des coins détectés dans l'image
            imgpoints.append(corners)

            # **3.4 Afficher les coins détectés pour validation visuelle**
            # - `cv2.drawChessboardCorners` dessine les coins détectés sur l'image pour vérifier leur exactitude.
            cv2.drawChessboardCorners(image, grid_size, corners, ret)
            cv2.imshow('Chessboard', image)  # Affiche l'image avec les coins détectés
            cv2.waitKey(500)  # Attend 500 ms avant de passer à l'image suivante

    # Fermer toutes les fenêtres d'affichage
    cv2.destroyAllWindows()

    # **4. Calibrer la caméra**
    # - `cv2.calibrateCamera` calcule les matrices intrinsèques et extrinsèques de la caméra.
    # - `objpoints` et `imgpoints` contiennent les correspondances entre les points 3D et leurs projections 2D.
    # - `gray.shape[::-1]` fournit la résolution de l'image utilisée pour le calibrage.
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # **5. Résultats**
    if ret:
        # Calibration réussie
        print("Calibration réussie !")
        print("\nMatrice intrinsèque :")
        print(camera_matrix)  # Matrice intrinsèque (focale, point principal)
        print("\nCoefficients de distorsion :")
        print(dist_coeffs)  # Distorsions radiales et tangentielle de l'objectif
    else:
        # Échec de la calibration
        print("Échec de la calibration.")

    # Retourner les résultats
    return camera_matrix, dist_coeffs, rvecs, tvecs


# **6. Configuration des images et paramètres**
# Liste des images d'échiquier (remplacez les chemins par ceux de vos images)
images = [
    '/path/to/image1.jpg',  # Remplacez avec vos images
    '/path/to/image2.jpg',
    '/path/to/image3.jpg'
]

# Taille de la grille de l'échiquier (colonnes, lignes de coins intérieurs)
# Par exemple, un échiquier 9x6 a 9 colonnes et 6 lignes de coins internes.
grid_size = (9, 6)

# Taille réelle des cases de l'échiquier (par exemple, en millimètres)
# Ex. : Si chaque case mesure 25 mm, utilisez `square_size = 25`.
square_size = 25  # 25 mm

# **7. Lancer la calibration**
# Calibrer la caméra avec les images fournies
camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(images, grid_size, square_size)
