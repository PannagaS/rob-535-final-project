import os
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
    plot_timeseries,
    generate_result_directory,
)
from mpc import simulate

### Parameters ###
image_name = "image3"
pos_goal = np.array([20, 28])
initial_heading = np.pi / 2


def main():
    # Read in image
    image = cv2.imread("./data/" + image_name + ".png")

    # Extract list of obstacles in image in pixel space
    obstacles_pixel_space = detect_vehicles(image)

    # Convert obstacles to world space
    ppm = 30 / 4
    ego_vehicle_x = 340
    ego_vehicle_y = 340
    obstacles_world_space = obstacles_to_world(
        obstacles=obstacles_pixel_space,
        ppm=ppm,
        x0=ego_vehicle_x,
        y0=ego_vehicle_y,
    )

    # Compute general form bounding ellipses
    ellipse_coefs = bounding_ellipse(obstacles_world_space)

    # Set initial parameters
    x0 = [0, 0, initial_heading, 0]
    parameters = np.concatenate((x0, pos_goal, np.array([0])))

    # Simulate vehicle trajectory under obstacle-aware NMPC policy
    xlog, ulog, tlog = simulate(ellipse_coefs, parameters)

    # Convert path to pixel coordinates
    path_log_pix = path_to_pix(xlog[:, :2], ppm, ego_vehicle_x, ego_vehicle_y)
    pos_goal_pix = path_to_pix(pos_goal, ppm, ego_vehicle_x, ego_vehicle_y)

    # Plot the results in world space
    world_plot = plot_path_world(xlog, pos_goal, ellipse_coefs)

    # Plot the path in the pixel space on the image
    bev_plot = plot_path_pix(image, path_log_pix, pos_goal_pix)

    # Plot control signals
    state_control_plot = plot_timeseries(tlog, xlog, ulog)

    # Save plots
    result_subdir = generate_result_directory(image_name)
    os.mkdir("./results/" + result_subdir)
    plt.figure(world_plot)
    plt.savefig(os.path.join("results", result_subdir, "world.png"))
    plt.figure(bev_plot)
    plt.savefig(os.path.join("results", result_subdir, "bev.png"))
    plt.figure(state_control_plot)
    plt.savefig(os.path.join("results", result_subdir, "state_control.png"))

    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
