import cv2
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from helpers import detect_vehicles, obstacles_to_world, bounding_ellipse, plot_ellipses
from mpc import simulate, plot_results


def main():
    # 1. Read in image
    image = cv2.imread("./data/easy_test.png")
    # 2. Extract list of obstacles in image in pixel space
    obstacles_pixel_space = detect_vehicles(image)
    # 3. Convert obstacles to world space
    obstacles_world_space = obstacles_to_world(
        obstacles=obstacles_pixel_space, ppm=15, x0=340, y0=340
    )
    # 4. Compute general form bounding ellipses
    ellipse_coefs = bounding_ellipse(obstacles_world_space)
    # Visualization
    # plt.figure()
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("Input Image")
    # plot_ellipses(ellipse_coefs)
    # 5. Run homework 2 clone code with additional obstacles

    # return
    parameters = [
        0,  # x position
        0,  # y position
        0,  # heading
        0,  # velocity
        4,  # x goal
        20,  # y goal
        5,  # v des
        0,  # delta_last
    ]

    xlog, ulog = simulate(ellipse_coefs, parameters)

    plot_results(xlog, ulog)


if __name__ == "__main__":
    main()
