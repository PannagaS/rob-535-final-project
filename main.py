import cv2
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from helpers import (
    detect_vehicles,
    obstacles_to_world,
    bounding_ellipse,
    path_to_pix,
    plot_path_world,
    plot_path_pix,
)
from mpc import simulate


def main():
    # 1. Read in image
    image = cv2.imread("./data/easy_test.png")
    # 2. Extract list of obstacles in image in pixel space
    obstacles_pixel_space = detect_vehicles(image)
    # 3. Convert obstacles to world space
    ppm = 30 / 4
    ego_vehicle_x = 340
    ego_vehicle_y = 340
    obstacles_world_space = obstacles_to_world(
        obstacles=obstacles_pixel_space,
        ppm=ppm,
        x0=ego_vehicle_x,
        y0=ego_vehicle_y,
    )
    # 4. Compute general form bounding ellipses
    ellipse_coefs = bounding_ellipse(obstacles_world_space)
    # Set initial parameters
    x0 = [0, 0, np.pi / 2, 0]
    x_goal = [4, 46]
    v_des = 5
    delta_last = 0
    parameters = x0 + x_goal + [v_des] + [delta_last]
    # Simulate vehicle trajectory under obstacle-aware NMPC policy
    xlog, ulog = simulate(ellipse_coefs, parameters)
    # Convert path to pixel coordinates
    path_log_pix = path_to_pix(xlog[:, :2], ppm, ego_vehicle_x, ego_vehicle_y)
    x_goal_pix = path_to_pix(
        np.asarray(x_goal).reshape((1, 2)), ppm, ego_vehicle_x, ego_vehicle_y
    )
    # Plot the results in world space
    plot_path_world(xlog, x_goal, ellipse_coefs)
    # Plot the path in the pixel space on the image
    plot_path_pix(image, path_log_pix, x_goal_pix)


if __name__ == "__main__":
    main()
