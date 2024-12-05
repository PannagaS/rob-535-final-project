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


def path_to_pix(path_log, ppm, x0, y0):
    """Convert a path (list of x-y coordinates) from the world frame to the pixel frame

    Args:
        path_log (ndarray): Nx2 array of world frame coordinates
        ppm (float): pixels per meter
        x0 (float): ego vehicle x position
        y0 (float): ego vehicle y position

    Returns:
        ndarray: Nx2 array of pixel frame coordinates
    """
    # Create homogenous coordinates
    path_log_h = np.hstack((path_log, np.ones((path_log.shape[0], 1))))
    # Transfer to pixel space
    path_log_pix = (world2pix(ppm, x0, y0) @ path_log_h.T).T
    # Convert to normal coordinates
    path_log_pix = path_log_pix[:, :2]
    # Return
    return path_log_pix


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
    rx, ry = np.sqrt(2) / 2 * 2 * size[0, :], np.sqrt(2) / 2 * 2 * size[1, :]
    angle = np.squeeze(angle)
    A = (np.cos(angle) / rx) ** 2 + (np.sin(angle) / ry) ** 2
    B = (np.sin(angle) / rx) ** 2 + (np.cos(angle) / ry) ** 2
    C = np.sin(2 * angle) * (1 / rx**2 - 1 / ry**2)
    D = -cy * C - 2 * cx * A
    E = -cx * C - 2 * cy * B
    F = A * cx**2 + B * cy**2 + C * cx * cy - 1
    return np.vstack((A, B, C, D, E, F)).T


def plot_ellipses(fig, coefs):
    """Helpful visualization tool, plots a set of ellipses gives a matrix of their general-from coefficients

    Args:
        coefs (coefs): Nx6 matrix where each row is the coefficients (A, B, C, D, E, F) for one ellipse
    """
    plt.figure(fig)
    for i in range(coefs.shape[0]):
        A, B, C, D, E, F = coefs[i, :]

        def ellipse(x, y):
            return A * x**2 + B * y**2 + C * x * y + D * x + E * y + F

        x = np.linspace(-50, 50, 401)
        y = np.linspace(-50, 50, 401)
        X, Y = np.meshgrid(x, y)

        Z = ellipse(X, Y)

        plt.contour(X, Y, Z, levels=[0], colors="black")

    plt.grid()
    plt.axis("equal")
    plt.title("Obstacles in World Coordinates")


def plot_path_world(xt, xg, ellipse_coeffs):
    """Plot path in world frame with ellipse obstacles

    Args:
        xt (ndarray): Nx4 array of states
        xg (ndarray, list): Goal point in world frame
        ellipse_coeffs (ndarray): Matrix of ellipse coefficients
    """
    x_w = xt[:, 0]
    y_w = xt[:, 1]

    fig = plt.figure(figsize=(8, 8))
    plt.plot(x_w, y_w, color="red", label="Path")
    plt.plot(
        x_w[0],
        y_w[0],
        color="lime",
        marker=".",
        markersize=10,
        markeredgecolor="k",
        label="Start",
    )
    plt.plot(
        xg[0],
        xg[1],
        color="yellow",
        marker="*",
        markersize=10,
        markeredgecolor="k",
        label="Goal",
    )
    plt.axis("scaled")
    plt.title("Trajectory")
    plt.legend()
    plt.xlabel("$x(m)$")
    plt.ylabel("$y(m)$")

    plot_ellipses(fig, ellipse_coeffs)


def plot_path_pix(image, path_log_pix, pos_goal_pix):
    """Plot the path on the image

    Args:
        image (image): Image to be plotted on
        path_log_pix (ndarray): Nx2 array of path points in pixels
        pos_goal_pix (ndarray, list): Goal point in pixels
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.plot(
        path_log_pix[:, 0], path_log_pix[:, 1], linewidth=2, color="red", label="Path"
    )
    plt.plot(
        path_log_pix[0, 0],
        path_log_pix[0, 1],
        color="lime",
        marker=".",
        markersize=10,
        markeredgecolor="k",
        label="Start",
    )
    plt.plot(
        pos_goal_pix[0, 0],
        pos_goal_pix[0, 1],
        color="yellow",
        marker="*",
        markersize=10,
        markeredgecolor="k",
        label="Goal",
    )
    plt.legend()
    plt.title("BEV Output with Planned Path")
    plt.axis("off")
    plt.show()
