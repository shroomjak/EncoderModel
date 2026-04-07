import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def random_unit_vector(size=3):
    v = np.random.normal(size=size)
    norm = np.linalg.norm(v)
    if norm == 0:
        return random_unit_vector(size)
    return v / norm

def get_angle(a, b, n):
    d = np.dot(a, b)
    vec_c = np.cross(a, b)
    c = np.linalg.norm(vec_c)
    s = np.sign(np.dot(vec_c, n))

    return np.atan2(s*c, d)

def random_in_plane_vector(n):
    norm_n = np.linalg.norm(n)
    if abs(norm_n - 1.0) > 1e-10:
        n = n / norm_n

    if abs(n[0]) < 0.9:
        a = np.array([1, 0, 0])
    elif abs(n[1]) < 0.9:
        a = np.array([0, 1, 0])
    else:
        a = np.array([0, 0, 1])

    u = np.cross(n, a)
    u = u / np.linalg.norm(u)

    v = np.cross(n, u)

    theta = np.random.uniform(0, 2 * np.pi)

    # Создаем случайный вектор в плоскости
    in_plane_vector = np.cos(theta) * u + np.sin(theta) * v

    return in_plane_vector

def main():
    # Rotation axis direction
    a = np.array([0, 0, 1], dtype=float)
    # Normal direction
    n0 = random_unit_vector()
    # Shift direction
    e0 = random_unit_vector() * 0.1
    # Read head direction
    u = np.array([0, 0, -1], dtype=float)
    # Read head position
    h = np.array([0, 1, 0], dtype=float)

    # Get zero-point direction
    l0 = random_in_plane_vector(n0)

    # Angle
    thetas = np.linspace(0, 2 * np.pi, 1000)
    theta_ests = []

    for theta in thetas:
        w = np.cos(theta / 2)
        x = np.sin(theta / 2) * a[0]
        y = np.sin(theta / 2) * a[1]
        z = np.sin(theta / 2) * a[2]
        r = R.from_quat(np.array([w, x, y, z]), scalar_first=True)

        # Rotate vectors
        e = r.apply(e0)
        n = r.apply(n0)
        l = r.apply(l0)

        # Calculate angle estimation
        i = h + np.dot(n, e - h) / np.dot(n, u) * u
        i_dir = (i - e) / np.linalg.norm(i - e)
        theta_est = get_angle(i_dir, l, n)
        theta_ests.append(theta_est)

    theta_ests = np.array(theta_ests)
    plt.plot(thetas, (theta_ests - thetas) % (2 * np.pi))
    plt.show()

if __name__ == '__main__':
    main()
