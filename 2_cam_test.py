import cv2

# Open the first camera (usually the webcam)
webcam = cv2.VideoCapture(0)

# Open the second camera (DroidCam or another external camera)
phone_camera = cv2.VideoCapture(1)

if not webcam.isOpened():
    print("Error: Cannot access webcam (ID: 0)")
    exit()

if not phone_camera.isOpened():
    print("Error: Cannot access DroidCam (ID: 1)")
    exit()

while True:
    # Capture frames from both cameras
    ret_webcam, frame_webcam = webcam.read()
    ret_phone, frame_phone = phone_camera.read()

    if not ret_webcam:
        print("Error: Cannot read frame from webcam.")
        break
    if not ret_phone:
        print("Error: Cannot read frame from DroidCam.")
        break

    # Display the frames
    cv2.imshow('Webcam', frame_webcam)
    cv2.imshow('Phone Camera (DroidCam)', frame_phone)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
phone_camera.release()
cv2.destroyAllWindows()
