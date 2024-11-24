import cv2
import numpy as np
import math

def gauss(x, y, sigma):
    """Calculate Gaussian value for coordinates (x, y) with standard deviation sigma."""
    part1 = 1 / (2 * math.pi * pow(sigma, 2))
    part2 = -(x * x + y * y) / (2 * pow(sigma, 2))
    return part1 * math.exp(part2)

def generate_gaussian_kernel(sigma=1.4, vois_mat=3):
    """Generate a Gaussian kernel matrix as a convolution mask."""
    vois = int(vois_mat / 2)
    kernel = []
    som = 0.0  # Sum of all values for normalization

    # Generate the Gaussian kernel values
    for i in range(-vois, vois + 1):
        row = []
        for j in range(-vois, vois + 1):
            val = gauss(i, j, sigma)
            row.append(val)
            som += val  # Accumulate the sum of kernel values for normalization
        kernel.append(row)

    # Normalize the kernel so that the sum of all elements is 1
    kernel = [[val / som for val in row] for row in kernel]
    kernel = np.array(kernel)  # Convert to NumPy array for easier matrix operations
    return kernel

def apply_convolution(image, kernel):
    """Apply convolution on the image using the provided Gaussian kernel."""
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2

    # Pad the image to handle borders
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    output = np.zeros_like(image)

    # Convolve the kernel over the image
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Apply the kernel on the region
            output[i, j] = np.sum(region * kernel)

    return output

# Load the image in grayscale
image = cv2.imread('house2.jpg', cv2.IMREAD_GRAYSCALE)

# Generate the Gaussian kernel
sigma = 1.4
vois_mat = 3
gaussian_kernel = generate_gaussian_kernel(sigma, vois_mat)
print("Generated Gaussian Convolution Kernel (normalized):")
print(gaussian_kernel)

# Apply Gaussian convolution
blurred_image = apply_convolution(image, gaussian_kernel)


# Display the original and blurred images
cv2.imshow("Original Image", image)
cv2.imshow("Gaussian Blurred Image", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
