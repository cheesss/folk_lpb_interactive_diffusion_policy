import numpy as np

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    """6D rotation: first two columns -> 3x3 matrix (columns are orthonormal)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-1)
    return out

def mat_to_rot6d(mat):
    """3x3 matrix -> 6D rotation (first two columns, column-major: col0 then col1)."""
    out = np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)
    return out


def random_rotation_matrix(rng=None):
    """Uniform random 3x3 rotation matrix (SO(3)): R^T R = I, det(R) = 1."""
    if rng is None:
        rng = np.random.default_rng()
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


a = random_rotation_matrix()
print(a)

rot6d_a = mat_to_rot6d(a)
print(rot6d_a)

reconstructed = rot6d_to_mat(rot6d_a)
print(reconstructed)