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


grad_magnitude = grad_magnitude.astype(np.uint8)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title('Gradient in x direction')
plt.imshow(grad_x, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Gradient in y direction')
plt.imshow(grad_y, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Gradient Magnitude')
plt.imshow(grad_magnitude, cmap='gray')

plt.tight_layout()
plt.show()
