import cv2
import numpy as np

# Update the HSV range for more precise red detection
lower_red1 = (0, 150, 100)  # Increase saturation and value minimum
upper_red1 = (10, 255, 255)
lower_red2 = (170, 150, 100)
upper_red2 = (180, 255, 255)

# Open a video stream (0 for default webcam, or use your smartphone's webcam feed URL)
# If you use an IP webcam app, replace '0' with the URL (e.g., 'http://192.168.x.x:8080/video').
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo.")
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for detecting red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # Mask for first red range
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # Mask for second red range
    mask = cv2.bitwise_or(mask1, mask2)  # Combine the two masks

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small objects
            # Get the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                print(f"Centre du point rouge : ({cx}, {cy})")

                # Draw a bounding rectangle around the red object
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw a circle at the center
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Display the original frame with detected objects
    cv2.imshow('Detected Red Objects', frame)

    # Display the mask for debugging purposes
    cv2.imshow('Red Mask', mask)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
