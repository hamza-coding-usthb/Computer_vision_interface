import cv2

# Load the image
image_path = 'test2.jpeg'
image = cv2.imread(image_path)

# Verify if the image is loaded successfully
if image is None:
    print("Failed to load the image.")
else:
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the red color range in HSV
    lower_red1 = (0, 120, 70)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 120, 70)
    upper_red2 = (180, 255, 255)

    # Create masks for the red color range
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours to draw borders and find centers
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Filter small contours
            # Get the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw the contour border
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Green border

                # Draw a point at the center
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)  # Blue point

                # Print the coordinates
                print(f"Red object center: ({cx}, {cy})")

    # Save the processed image to visualize the result
    output_path = 'test2_detected.jpg'
    cv2.imwrite(output_path, image)
    print(f"Processed image saved at: {output_path}")
