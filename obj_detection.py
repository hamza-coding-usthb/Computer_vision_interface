import cv2

# Charger l'image
image_path = 'red.png'  # Chemin de l'image à analyser
image = cv2.imread(image_path)  # Lire l'image dans une matrice OpenCV (BGR)

# Vérifier si l'image est chargée correctement
if image is None:
    print("Erreur : Impossible de charger l'image. Vérifiez le chemin du fichier.")
else:
    # Convertir l'image en espace de couleur HSV
    # Pourquoi HSV ? L'espace HSV (Hue, Saturation, Value) facilite la détection des couleurs, car il sépare :
    # - La teinte (Hue) : Qui représente la couleur principale (rouge, vert, bleu, etc.).
    # - La saturation : Qui mesure l'intensité ou la pureté de la couleur.
    # - La valeur : Qui correspond à la luminosité.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir les plages de couleurs pour le rouge
    # Le rouge est particulier car il apparaît à deux extrémités de la roue chromatique (autour de 0° et 360° en HSV).
    lower_red1 = (0, 120, 70)  # Plage pour le rouge vif (proche de 0°)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 120, 70)  # Plage pour l'autre extrémité du rouge (proche de 360°)
    upper_red2 = (180, 255, 255)

    # Créer des masques pour détecter les pixels rouges
    # cv2.inRange : Cette fonction crée un masque binaire (noir et blanc) :
    # - Les pixels qui se trouvent dans la plage définie (rouge ici) deviennent blancs (255).
    # - Les pixels en dehors de la plage deviennent noirs (0).
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # Rouge vif
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # Rouge sombre
    mask = cv2.bitwise_or(mask1, mask2)  # Combine les deux masques pour détecter toutes les nuances de rouge

    # Nettoyer le masque avec des opérations morphologiques
    # Pourquoi nettoyer ? Le masque peut contenir du bruit (petits pixels blancs isolés ou trous dans les objets rouges).
    # La fermeture (cv2.MORPH_CLOSE) remplit les petits trous à l'intérieur des objets et connecte les parties fragmentées.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Élément structurant elliptique de 5x5
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Appliquer la fermeture

    # Trouver les contours dans le masque
    # cv2.findContours : Trouve les bords (contours) des objets blancs dans le masque.
    # - cv2.RETR_TREE : Récupère tous les contours et établit une hiérarchie parent-enfant (utile pour des objets imbriqués).
    # - cv2.CHAIN_APPROX_SIMPLE : Simplifie les contours en supprimant les points redondants (par exemple, une ligne droite est réduite à ses extrémités).
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Vérifier si des contours ont été détectés
    if len(contours) == 0:
        print("Aucun point rouge détecté.")
    else:
        # Parcourir tous les contours détectés
        for contour in contours:
            # Calculer l'aire du contour
            area = cv2.contourArea(contour)  # Calcule la surface du contour en pixels
            if area > 10:  # Filtrer les petits objets (éviter les pixels parasites ou le bruit)
                # Obtenir les moments du contour pour trouver son centre
                # cv2.moments : Calcule les moments géométriques d'un contour.
                # Ces moments sont utilisés pour trouver le centre (cx, cy) de l'objet.
                M = cv2.moments(contour)
                if M["m00"] != 0:  # Éviter la division par zéro (si le contour a une aire nulle)
                    cx = int(M["m10"] / M["m00"])  # Coordonnée X du centre
                    cy = int(M["m01"] / M["m00"])  # Coordonnée Y du centre
                    print(f"Centre du point rouge : ({cx}, {cy})")  # Affiche les coordonnées du centre

                    # Dessiner un rectangle englobant autour du contour
                    # cv2.boundingRect : Retourne un rectangle minimal qui encadre entièrement le contour.
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectangle vert autour de l'objet

                    # Dessiner un point au centre
                    # cv2.circle : Dessine un cercle, ici pour marquer le centre de l'objet.
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)  # Cercle bleu au centre

        # Afficher les masques pour analyse
        # Ces fenêtres montrent comment chaque masque isole les pixels rouges.
        cv2.imshow('Mask1', mask1)  # Masque pour la première plage de rouge
        cv2.imshow('Mask2', mask2)  # Masque pour la seconde plage de rouge
        cv2.imshow('Final Mask', mask)  # Masque final combiné

        # Afficher l'image avec les rectangles et les points
        cv2.imshow('Detected Red Dots', image)

        # Attendre une touche pour fermer les fenêtres
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Sauvegarder l'image traitée
        output_path = 'red_dot_output.png'
        cv2.imwrite(output_path, image)  # Enregistre l'image avec les rectangles et points
        print(f"Image traitée sauvegardée sous : {output_path}")
