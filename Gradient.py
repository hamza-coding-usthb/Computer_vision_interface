import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('istockphoto-513247652-612x612.jpg', cv2.IMREAD_GRAYSCALE)


grad_x = img[:, :img.shape[1] - 1] - img[:, 1:]


grad_y = img[:img.shape[0] - 1, :] - img[1:img.shape[0], :]


common_height = min(grad_x.shape[0], grad_y.shape[0])
common_width = min(grad_x.shape[1], grad_y.shape[1])


grad_x = grad_x[:common_height, :common_width]
grad_y = grad_y[:common_height, :common_width]


grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
grad_magnitude = grad_magnitude.astype(np.uint8)  # Convert to uint8 for display


th = 0
type_th = 0

"""def afficher2():
    # Apply different thresholding types based on `type_th` using OpenCV's predefined functions
    if type_th == 0:  # Binary threshold
        _, imgRes = cv2.threshold(grad_magnitude, th, 255, cv2.THRESH_BINARY)
    elif type_th == 1:  # Binary Inverse threshold
        _, imgRes = cv2.threshold(grad_magnitude, th, 255, cv2.THRESH_BINARY_INV)
    elif type_th == 2:  # Truncate threshold
        _, imgRes = cv2.threshold(grad_magnitude, th, 255, cv2.THRESH_TRUNC)
    elif type_th == 3:  # To Zero threshold
        _, imgRes = cv2.threshold(grad_magnitude, th, 255, cv2.THRESH_TOZERO)
    elif type_th == 4:  # To Zero Inverse threshold
        _, imgRes = cv2.threshold(grad_magnitude, th, 255, cv2.THRESH_TOZERO_INV)
    
    cv2.imshow("Thresholded Gradient Magnitude", imgRes)
"""
def afficher():
    imgRes = np.zeros_like(grad_magnitude)  # Create an output image of the same size as grad_magnitude
    sup_th = grad_magnitude > th
    inf_th = np.invert(sup_th)
    
    if (type_th == 0):
        imgRes[sup_th] = 255
        imgRes[inf_th] = 0
    elif (type_th == 1):
        imgRes[sup_th] = 0
        imgRes[inf_th] = 255
    elif (type_th == 2):
        imgRes[sup_th] = th
    elif (type_th == 3):
        imgRes[inf_th] = 0
    elif (type_th == 4):
        imgRes[inf_th] = grad_magnitude[inf_th]
    
    cv2.imshow("Thresholded Gradient Magnitude", imgRes)
# Functions for trackbar changes
def change_th(x):
    global th
    th = x
    afficher()

def change_type(x):
    global type_th
    type_th = x
    afficher()

# Initial display
afficher()


cv2.createTrackbar("thresh", "Thresholded Gradient Magnitude", 0, 255, change_th)
cv2.createTrackbar("type", "Thresholded Gradient Magnitude", 0, 4, change_type)
cv2.waitKey(0)
cv2.destroyAllWindows()
