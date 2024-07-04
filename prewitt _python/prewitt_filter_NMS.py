import numpy as np
import cv2
from tkinter import filedialog
import os
import matplotlib.pyplot as plt

def prw_edgedet():
    # Prewitt operator kernels
    prw_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Prewitt Gx
    prw_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # Prewitt Gy

    # Load image
    filename = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
    if not filename:
        print("No file selected. Exiting.")
        return
    
    im = cv2.imread(filename)
    if im is None:
        print(f"Failed to load image: {filename}")
        return

    # Convert to grayscale if necessary
    if len(im.shape) == 3:
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray_im = im

    # Apply Gaussian smoothing
    sigma = 1.0  # Standard deviation of the Gaussian distribution
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # Kernel size (odd number)
    smoothed_im = cv2.GaussianBlur(gray_im, (kernel_size, kernel_size), sigma)

    # Convolve image with Prewitt kernels to get IX and IY arrays
    edge_x = cv2.filter2D(smoothed_im.astype(float), -1, prw_x)
    edge_y = cv2.filter2D(smoothed_im.astype(float), -1, prw_y)

    # Calculate gradient magnitude array from IX and IY arrays
    gradient_mag = np.sqrt(edge_x ** 2 + edge_y ** 2)

    # Calculate gradient direction
    gradient_dir = np.arctan2(edge_y, edge_x) * 180 / np.pi
    gradient_dir[gradient_dir < 0] += 180  # Ensure gradient direction is within [0, 180] degrees

    # Non-maximum suppression to thin the edges
    nms = non_maxima_suppress_custom(gradient_mag, gradient_dir)

    # Apply thresholding
    threshold = 30  # Adjust this value as needed
    nms[nms < threshold] = 0

    # Display final result after NMS and thresholding
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(gray_im, cmap='gray'), plt.title('Original')
    plt.subplot(132), plt.imshow(smoothed_im, cmap='gray'), plt.title('Smoothed')
    plt.subplot(133), plt.imshow(nms.astype(np.uint8), cmap='gray'), plt.title('Final result')
    plt.tight_layout()
    plt.show()

    # Save the output image
    output_filename = os.path.splitext(filename)[0] + '_output.png'
    cv2.imwrite(output_filename, nms.astype(np.uint8))
    print(f"Output image saved as: {output_filename}")

def non_maxima_suppress_custom(im, angle):
    nms = np.zeros_like(im)
    for y in range(1, im.shape[0] - 1):
        for x in range(1, im.shape[1] - 1):
            if (0 <= angle[y, x] < 22.5) or (157.5 <= angle[y, x] <= 180):
                neighbour_value = max(im[y, x-1], im[y, x+1])
            elif 22.5 <= angle[y, x] < 67.5:
                neighbour_value = max(im[y-1, x-1], im[y+1, x+1])
            elif 67.5 <= angle[y, x] < 112.5:
                neighbour_value = max(im[y-1, x], im[y+1, x])
            else:
                neighbour_value = max(im[y-1, x+1], im[y+1, x-1])

            if im[y, x] < neighbour_value:
                nms[y, x] = 0
            else:
                nms[y, x] = im[y, x]
    return nms

if __name__ == "__main__":
    prw_edgedet()