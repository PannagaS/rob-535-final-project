import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_vehicles(image):
    """Detect vehicles in input image

    Args:
        image (string): file name of target input image

    Returns:
        tuple: (centers, sizes, angles, number)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color
    lower_blue = np.array([102, 174, 82])
    upper_blue = np.array([121, 255, 255])

    # Create a mask for blue areas
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Filter the image using the mask to isolate blue regions
    res = cv2.bitwise_and(image, image, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize np arrays for centers, sizes, and angles
    centers = np.empty((2, 0))
    sizes = np.empty((2, 0))
    angles = np.empty((0))

    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if it is a rectangle (4 sides and convex)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Calculate the area and filter based on minimum area
            area = cv2.contourArea(contour)

            if area >= 400:
                # Calculate bounding box of obstacle
                bounding_box = cv2.minAreaRect(contour)
                # Extract center, size, and angle of box
                center, size, angle = bounding_box
                # Record in lists
                centers = np.append(
                    centers, np.array([center[0], center[1]]).reshape((2, 1)), axis=1
                )
                sizes = np.append(
                    sizes, np.array([size[0], size[1]]).reshape((2, 1)), axis=1
                )
                angles = np.append(angles, angle)

    num_obs = centers.shape[1]
    rectangles = (centers, sizes, angles, num_obs)

    return rectangles


def world2pix(ppm, x0, y0):
    """Returns homogenous transformation matrix which turns world coordinates into pixel coordinates

    Args:
        ppm (double): pixels per meter
        x0 (double): ego car x-coordinate in pixels
        y0 (double): ego car y-coordinate in pixels

    Returns:
        Twp: Homogenous transformation matrix which maps world coordinates to pixel coordinates
    """
    S = np.array([[ppm, 0, 0], [0, ppm, 0], [0, 0, 1]])
    Hwp = np.array([[1, 0, x0], [0, -1, y0], [0, 0, 1]])
    return Hwp @ S


def pix2world(ppm, x0, y0):
    """Compute homogenous transformation matrix which turns pixels coordinates into world coordinates

    Args:
        ppm (double): pixels per meter
        x0 (double): ego car x-coordinate in pixels
        y0 (double): ego car y-coordinate in pixels

    Returns:
        Tpw: Homogenous tranformation matrix which turns pixel coordinates into world coordinates
    """
    Sinv = np.array([[1 / ppm, 0, 0], [0, 1 / ppm, 0], [0, 0, 1]])
    Hpw = np.array([[1, 0, -x0], [0, -1, y0], [0, 0, 1]])
    return Sinv @ Hpw


def obstacles_to_world(obstacles, ppm, x0, y0):
    """Convert list of obstacles in pixels space to world space

    Args:
        obstacles (list): List of tuples (center, size, angle)
        ppm (float): pixels per meter
        x0 (float): ego car x coordinate in pixels
        y0 (float): ego car y coordinate in pixels
    """
    T_pix_2_world = pix2world(ppm=ppm, x0=x0, y0=y0)
    # Extract obstacles characteristics
    centers_pix, sizes_pix, angles_pix, num_obs = obstacles
    # Convert center coordinates
    centers_world = (
        T_pix_2_world
        @ np.vstack(
            (
                centers_pix,
                np.ones((1, num_obs)),
            )
        )
    )[:2, :]
    # Convert sizes
    sizes_world = sizes_pix / ppm
    # Convert angles
    angles_world = -np.deg2rad(angles_pix)
    # Repackage and return
    return (centers_world, sizes_world, angles_world, num_obs)


def bounding_ellipse(rectangles):
    """Given a rectangle, computes the coefficients of the general form of the smallest ellipse which both (1) shares the  same aspect ratio as the box, and (2) encompasses the box entirely

    Args:
        rectangles (tuple): (center, size, angle), note that this function is vectorized to work on ndarrays

    Returns:
        np.ndarray: Matrix [A, B, C, D, E, F] where each row corresponds to a different ellipse and each column to
        a coefficient
    """
    center, size, angle, _ = rectangles
    cx, cy = center[0, :], center[1, :]
    rx, ry = np.sqrt(2) / 2 * size[0, :], np.sqrt(2) / 2 * size[1, :]
    angle = np.squeeze(angle)
    A = (np.cos(angle) / rx) ** 2 + (np.sin(angle) / ry) ** 2
    B = (np.sin(angle) / rx) ** 2 + (np.cos(angle) / ry) ** 2
    C = np.sin(2 * angle) * (1 / rx**2 - 1 / ry**2)
    D = -cy * C - 2 * cx * A
    E = -cx * C - 2 * cy * B
    F = A * cx**2 + B * cy**2 + C * cx * cy - 1
    return np.vstack((A, B, C, D, E, F)).T


def plot_ellipses(coefs):
    """Helpful visualization tool, plots a set of ellipses gives a matrix of their general-from coefficients

    Args:
        coefs (coefs): Nx6 matrix where each row is the coefficients (A, B, C, D, E, F) for one ellipse
    """
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[0]):
        A, B, C, D, E, F = coefs[i, :]
        # Define Q matrix and p vector
        Q = np.array([[A, C / 2], [C / 2, B]])
        p = np.array([D, E])

        # Solve for the center
        center = -0.5 * np.linalg.solve(Q, p)
        x_c, y_c = center

        # Eigen decomposition for orientation and axes
        eigenvalues, eigenvectors = np.linalg.eig(Q)
        axis_lengths = np.sqrt(1 / np.abs(eigenvalues))  # Semi-axis lengths
        semi_major, semi_minor = np.max(axis_lengths), np.min(axis_lengths)

        # Orientation (angle of rotation)
        col = 0 if eigenvalues[0] < eigenvalues[1] else 1
        angle = np.arctan2(eigenvectors[1, col], eigenvectors[0, col]) - np.pi
        # Generate ellipse points
        theta = np.linspace(0, 2 * np.pi, 500)
        x_ellipse = semi_major * np.cos(theta)
        y_ellipse = semi_minor * np.sin(theta)

        # Rotate and translate ellipse
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ellipse = R @ np.array([x_ellipse, y_ellipse])
        x_ellipse, y_ellipse = ellipse[0, :] + x_c, ellipse[1, :] + y_c

        # Plot
        plt.plot(x_ellipse, y_ellipse, label=f"Obstacle {i + 1}")
        plt.scatter(x_c, y_c, color="red")

    plt.grid()
    plt.axis("equal")
    plt.legend()
    plt.title("Obstacles in World Coordinates")
    plt.show()
