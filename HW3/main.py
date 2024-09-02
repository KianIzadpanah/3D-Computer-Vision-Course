import numpy as np


def cart_2_hom(point):
    return np.append(point, [1])


def rotation_matrix(theta, omega):
    theta = np.radians(theta)
    omega = omega / np.linalg.norm(omega)
    omega_hat = np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * omega_hat + \
        (1 - np.cos(theta)) * (omega_hat @ omega_hat)
    return R


def similarity_transformation(theta, scale, translation, axis):
    R = rotation_matrix(theta, axis)
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = translation
    return T


def apply_transformation(transform, points):
    transformed_points = transform @ points.T
    return transformed_points


def test_transformations(theta, scale_factor, translation_vector, omega, point):
    points_homogeneous = cart_2_hom(point)
    transform_matrix = similarity_transformation(
        theta, scale_factor, translation_vector, omega)
    return apply_transformation(transform_matrix, points_homogeneous)


def inverse_transform(theta, translation_vector, omega, transformed_point):
    transformed_points_homogeneous = cart_2_hom(transformed_point)
    transform_matrix = similarity_transformation(
        theta, scale_factor, translation_vector, omega)
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    initial_point = inv_transform_matrix @ transformed_points_homogeneous.T
    return initial_point


# let's test the third part of the forth theory question
theta = 90
scale_factor = 10,
translation_vector = np.array([-8, 1, 0])
omega = np.array([3, 5, -6])
point = np.array([2, -7, 9])
transformed_point = test_transformations(
    theta, scale_factor, translation_vector, omega, point)
print("The result of the similarity transformation on point=(2, -7, 9) with Theta=90 degree, scale factor=10, translation vector=(-8, 1, 0), around the vector=(3, 5, -6) is:\n", transformed_point.T)


# Now lets calculate the initial point which resulted in (-3, 6, -1) after the previous mentioned transformation
transformed_point = np.array([-3, 6, -1])
initial_point = inverse_transform(
    theta, translation_vector, omega, transformed_point)
print("The initial point is:\n", initial_point.T)
