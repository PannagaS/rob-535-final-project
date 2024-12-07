import cv2
import numpy as np


image = cv2.imread("data/3e8750f331d7499e9b5123e9eb70f2e2_bev.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500:  # Adjust the area threshold as needed to isolate the "X"
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"Center of X: ({cX}, {cY})")
            cv2.circle(image, (cX, cY), 3, (0, 0, 255), -1)


cv2.imshow("Image with Center of X", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
