import numpy as np
import cv2
import matplotlib.pyplot as plt

from IPython import embed

# Load the image
img = cv2.imread("data/3e8750f331d7499e9b5123e9eb70f2e2_bev.png")
cv2.imshow("Original Image", img)

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for blue color
lower_blue = np.array([102, 174, 82])
upper_blue = np.array([121, 255, 255])

# Create a mask for blue areas
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Filter the image using the mask to isolate blue regions
res = cv2.bitwise_and(img, img, mask=mask)

# Function to detect rectangles and lines
def detect_rectangles(image, min_area):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list to store rectangles
    rectangles = []


    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it is a rectangle (4 sides and convex)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Calculate the area and filter based on minimum area
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(approx)
                rectangles.append((approx, (x, y, w, h)))


    return rectangles


# Detect rectangles
min_area = 400  # Adjusted minimum area to ensure meaningful rectangles are found
rectangles = detect_rectangles(res, min_area)

# Draw detected rectangles on the original image

for rectangle, (x,y,w,h) in rectangles:
    cv2.drawContours(img, [rectangle], -1, (0, 0, 255), 3)  # Draw rectangles in red color
    # print("Location of car: ", x + w//2, y + w//2)

# embed()
height, width = img.shape[:2]
rect_width = 19  # Width of the rectangle
rect_height = 35  # Height of the rectangle

# Calculate the top-left and bottom-right points of the rectangle
center_x = width // 2
center_y = height // 2
x_offset = 11
y_offset = 5
top_left = (x_offset + center_x - rect_width // 2, y_offset + center_y - rect_height // 2)
bottom_right = (x_offset + center_x + rect_width // 2, y_offset + center_y + rect_height // 2)
cv2.rectangle(img,  top_left, bottom_right, (200,100,40), 3)



cv2.imshow("Detected Blue Rectangles", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
