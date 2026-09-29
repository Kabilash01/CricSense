"""Full 3D camera per frame, recovered from the ground-plane homography.

The pitch homography maps the ground (z = 0) to the image. Assuming square pixels, no
skew and the principal point at the image centre - standard for broadcast cameras - the
focal length follows from the homography alone (Zhang's orthogonality constraints), and
with it the complete projection. That is what lets anything above the ground be placed:
the ball in flight, the tops of the stumps.

World frame: pitch metres (x along the pitch from the striker's stumps, y across it) with
z pointing up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PoseError(ValueError):
    pass


@dataclass(frozen=True)
class CameraPose:
    P: np.ndarray  # 3x4 projection, world metres (x, y, z up) -> homogeneous pixels
    focal_px: float
    centre: np.ndarray  # camera position in world metres

    @property
    def K(self) -> np.ndarray:
        cx, cy = self.principal_point
        f = self.focal_px
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])

    @property
    def principal_point(self) -> tuple[float, float]:
        # recovered from P: the principal point is where the optical axis meets the image
        M = self.P[:, :3]
        m3 = M[2] / np.linalg.norm(M[2])
        scale = np.linalg.norm(M[2])
        return float(M[0] @ m3 / scale), float(M[1] @ m3 / scale)

    @property
    def R(self) -> np.ndarray:
        """World -> camera rotation."""
        R = np.linalg.solve(self.K, self.P[:, :3])
        return R / np.cbrt(np.linalg.det(R))

    @classmethod
    def from_krc(cls, K: np.ndarray, R: np.ndarray, centre: np.ndarray) -> CameraPose:
        P = K @ np.c_[R, -R @ centre]
        if (P[2, :3] @ (np.array([10.0, 0.0, 0.0]) - centre)) < 0:
            P = -P
        return cls(P=P / np.linalg.norm(P[2, :3]), focal_px=float(K[0, 0]), centre=np.asarray(centre, float).copy())

    def image_to_pitch(self) -> np.ndarray:
        """The ground-plane (z = 0) homography this camera implies: pixels -> metres."""
        H = np.linalg.inv(self.P[:, [0, 1, 3]])
        return H / H[2, 2]

    @classmethod
    def from_homography(cls, image_to_pitch: np.ndarray, image_size: tuple[int, int]) -> CameraPose:
        w, h = image_size
        cx, cy = w / 2.0, h / 2.0
        G = np.linalg.inv(np.asarray(image_to_pitch, dtype=np.float64))  # pitch (x, y, 1) -> image
        T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
        Hc = T @ G  # principal point at the origin
        h1, h2 = Hc[:, 0], Hc[:, 1]
        # r1 . r2 = 0 and |r1| = |r2|, each linear in w = 1/f^2: a*w + b = 0
        a1, b1 = h1[0] * h2[0] + h1[1] * h2[1], h1[2] * h2[2]
        a2, b2 = h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2, h1[2] ** 2 - h2[2] ** 2
        denom = a1 * a1 + a2 * a2
        if denom <= 0:
            raise PoseError("degenerate homography: focal length is unobservable")
        inv_f2 = -(a1 * b1 + a2 * b2) / denom
        if inv_f2 <= 0:
            raise PoseError("homography is inconsistent with a real camera (no positive focal length)")
        f = 1.0 / np.sqrt(inv_f2)

        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        B = np.linalg.inv(K) @ G
        scale = 2.0 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
        r1, r2, t = B[:, 0] * scale, B[:, 1] * scale, B[:, 2] * scale
        # Nearest rotation to (r1, r2, r1 x r2); the sign of the third axis is chosen below.
        U, _, Vt = np.linalg.svd(np.stack([r1, r2, np.cross(r1, r2)], axis=1))
        R = U @ Vt
        r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
        if t[2] < 0:  # the pitch must be in front of the camera
            r1, r2, t = -r1, -r2, -t
            r3 = np.cross(r1, r2)
        M = K @ np.stack([r1, r2, r3], axis=1)
        centre = -np.linalg.solve(M, K @ t)
        if centre[2] < 0:  # z must point up: the camera is above the ground
            r3 = -r3
            M = K @ np.stack([r1, r2, r3], axis=1)
            centre = -np.linalg.solve(M, K @ t)
        P = np.concatenate([M, (K @ t)[:, None]], axis=1)
        closed_form = cls(P=P / np.linalg.norm(P[2, :3]), focal_px=float(f), centre=centre)
        return closed_form._refined(image_to_pitch, image_size)

    def _refined(self, image_to_pitch: np.ndarray, image_size: tuple[int, int]) -> CameraPose:
        """Best physical camera for the homography, starting from the closed-form one.

        The closed form forces an orthonormal rotation onto a homography estimated from a
        few noisy clicks; with a telephoto lens that can move the ground projection by
        100+ px even though the homography itself is good to ~1 px. Refining focal length,
        rotation and position (7 numbers) against ground points around the pitch brings the
        camera back to the homography's accuracy.
        """
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        ground, pix = visible_ground_samples(image_to_pitch, image_size)
        if len(ground) < 8:
            return self
        w, h = image_size
        cx, cy = w / 2.0, h / 2.0
        pts = np.c_[ground, np.zeros(len(ground))]
        R0 = self.R
        U, _, Vt = np.linalg.svd(R0)
        x0 = np.r_[self.focal_px, Rotation.from_matrix(U @ Vt).as_rotvec(), self.centre]

        def residual(q: np.ndarray) -> np.ndarray:
            K = np.array([[q[0], 0, cx], [0, q[0], cy], [0, 0, 1.0]])
            R = Rotation.from_rotvec(q[1:4]).as_matrix()
            p = (pts - q[4:7]) @ R.T @ K.T
            return (p[:, :2] / p[:, 2:3] - pix).ravel()

        fit = least_squares(residual, x0, method="lm")
        if not fit.success or fit.x[0] <= 0 or fit.x[6] <= 0:
            return self
        K = np.array([[fit.x[0], 0, cx], [0, fit.x[0], cy], [0, 0, 1.0]])
        return CameraPose.from_krc(K, Rotation.from_rotvec(fit.x[1:4]).as_matrix(), fit.x[4:7])

    def with_centre(self, image_to_pitch: np.ndarray, image_size: tuple[int, int], centre: np.ndarray) -> CameraPose:
        """This frame's camera with its position fixed: fit only focal length and rotation.

        A broadcast camera turns and zooms but does not move. Fitting each frame's position
        freely lets it wander by tens of metres during a zoom (measured: 67-106 m within one
        delivery), and then the sight lines to a flying ball from different frames no longer
        meet. With the position fixed at the view's reference camera, 4 parameters are fitted
        to this frame's own ground calibration around the pitch. `self` seeds the fit.
        """
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        ground, pix = visible_ground_samples(image_to_pitch, image_size)
        if len(ground) < 6:
            raise PoseError("too little of the pitch visible to fit the camera")
        w, h = image_size
        cx, cy = w / 2.0, h / 2.0
        pts = np.c_[ground, np.zeros(len(ground))]
        C = np.asarray(centre, dtype=np.float64)
        U, _, Vt = np.linalg.svd(self.R)

        def residual(q: np.ndarray) -> np.ndarray:
            K = np.array([[q[0], 0, cx], [0, q[0], cy], [0, 0, 1.0]])
            p = (pts - C) @ Rotation.from_rotvec(q[1:4]).as_matrix().T @ K.T
            return (p[:, :2] / p[:, 2:3] - pix).ravel()

        fit = least_squares(residual, np.r_[self.focal_px, Rotation.from_matrix(U @ Vt).as_rotvec()], method="lm")
        if not fit.success or fit.x[0] <= 0:
            raise PoseError("fixed-position camera fit did not converge")
        K = np.array([[fit.x[0], 0, cx], [0, fit.x[0], cy], [0, 0, 1.0]])
        return CameraPose.from_krc(K, Rotation.from_rotvec(fit.x[1:4]).as_matrix(), C)

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """(N, 3) world metres -> (N, 2) pixels."""
        pts = np.c_[np.asarray(xyz, dtype=np.float64).reshape(-1, 3), np.ones(len(np.asarray(xyz).reshape(-1, 3)))]
        p = pts @ self.P.T
        return p[:, :2] / p[:, 2:3]

    def ray(self, uv: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """The world-space ray through a pixel: (origin = camera centre, unit direction)."""
        M = self.P[:, :3]
        d = np.linalg.solve(M, np.array([uv[0], uv[1], 1.0]))
        d = d / np.linalg.norm(d)
        # point the ray away from the camera, toward the scene in front of it
        if (self.P[2, :3] @ d) < 0:
            d = -d
        return self.centre.copy(), d


def visible_ground_samples(
    image_to_pitch: np.ndarray, image_size: tuple[int, int], n: int = 15, bottom_margin: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Ground points (metres) and their pixels, sampled over the visible part of the frame.

    Pixels are spread across the image (above the score bar) and mapped to the ground; ones
    at or beyond the horizon, or implausibly far from the pitch, are dropped. Fitting and
    scoring on these - rather than on fixed pitch points that a zoomed view may not even
    show - measures the camera where the image actually constrains it.
    """
    w, h = image_size
    us = np.linspace(0.02 * w, 0.98 * w, n)
    vs = np.linspace(0.02 * h, (1 - bottom_margin) * h, n)
    pix = np.array([(u, v) for u in us for v in vs])
    homog = np.c_[pix, np.ones(len(pix))] @ np.asarray(image_to_pitch, dtype=np.float64).T
    ok = homog[:, 2] > 1e-9 if np.median(homog[:, 2]) > 0 else homog[:, 2] < -1e-9
    ground = homog[:, :2] / homog[:, 2:3]
    # The homography is fitted to crease corners, so it is trustworthy on and around the
    # pitch and pure extrapolation out in the field: keep only samples near the pitch.
    ok &= (ground[:, 0] > -8) & (ground[:, 0] < 28) & (np.abs(ground[:, 1]) < 8)
    return ground[ok], pix[ok]


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]
