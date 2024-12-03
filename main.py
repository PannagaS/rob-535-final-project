import cv2
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from helpers import detect_vehicles, obstacles_to_world, bounding_ellipse, plot_ellipses
from mpc import nmpc_controller


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
    prob, N_mpc, n_x, n_g, n_p, lb_var, ub_var, lb_cons, ub_cons = nmpc_controller(
        ellipse_coefs
    )
    opts = {"ipopt.print_level": 3, "print_time": 0}
    solver = ca.nlpsol("solver", "ipopt", prob, opts)


if __name__ == "__main__":
    main()
