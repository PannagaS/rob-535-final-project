import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def filter_by_blue(image):
    # Return binary image where strongly blue pixels are white and all other pixels are black
    b, g, r = cv.split(image)
    return 255 * np.uint8(np.bitwise_and(b >= 200, g < 100, r < 100))


def filter_by_blob_size(image):
    # New image which will only have the large blobs on it
    large_blob_image = np.zeros_like(image, dtype=np.uint8)
    # Experimentally determined threshold
    min_area = 400
    # Find all blobs
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Loop through all contours...
    for contour in contours:
        # Get the bounding box
        bounding_box = cv.minAreaRect(contour)
        # Get the dimensions of the bounding box
        _, size, _ = bounding_box
        # Compute aread
        area = size[0] * size[1]
        # If obstacle is sufficiently large, paint its bounding box onto the new image in white
        if area > min_area:
            cv.fillPoly(large_blob_image, [np.int32(cv.boxPoints(bounding_box))], 255)
    return large_blob_image


def main():
    image = cv.imread("./data/easy_test.png")
    cropped = image[120:800, 120:800, :]
    binary_image = filter_by_blue(cropped)
    car_image = filter_by_blob_size(binary_image)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(cropped, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cv.cvtColor(binary_image, cv.COLOR_GRAY2RGB))
    plt.title("Filtered by Color")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(car_image, cv.COLOR_GRAY2RGB))
    plt.title("Filtered by Blob Size")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
