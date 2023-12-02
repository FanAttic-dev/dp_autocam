from functools import cached_property
import cv2
import numpy as np
from camera.camera import Camera
from utils.config import Config
from utils.constants import DT_FLOAT, DT_INT, INTERPOLATION_TYPE, Color, DrawingMode
import utils.utils as utils


class RectilinearCamera(Camera):
    SENSOR_DX = 10
    FOV_DX = 5
    TILT_DX = 5

    def __init__(
        self,
        frame_orig,
        config: Config,
        ignore_bounds=Config.autocam["debug"]["ignore_bounds"]
    ):
        self.lens_fov_horiz_deg = config.dataset["camera_params"]["lens_fov_horiz_deg"]
        self.pitch_tilt_deg = config.dataset["camera_params"]["pitch_tilt_deg"]
        super().__init__(frame_orig, config, ignore_bounds)

    # region FoV
    @property  # could use @cached_property for optimization
    def lens_fov_vert_deg(self):
        return self.lens_fov_horiz_deg / Camera.FRAME_ASPECT_RATIO
        # return utils.hFoV2vFoV(self.lens_fov_horiz_deg, Camera.FRAME_ASPECT_RATIO)

    @property
    def fov_rad(self):
        return np.deg2rad(
            np.array([self.fov_horiz_deg, self.fov_vert_deg]),
            dtype=DT_FLOAT
        )

    @property  # could use @cached_property for optimization
    def limits(self):
        limits = np.array(
            [self.lens_fov_horiz_deg, self.lens_fov_vert_deg],
            dtype=DT_FLOAT
        ) / 2
        return np.deg2rad(limits)

    def screen_width_px2fov(self, px):
        """Calculate the field of view given by width in screen space [px]."""
        frame_orig_width, _ = self.frame_orig_size
        return self.lens_fov_horiz_deg / frame_orig_width * px
    # endregion

    def _screen2spherical(self, coord_screen):
        """ In range: [0, 1], out range: [-FoV_lens/2, FoV_lens/2] """
        return (coord_screen * 2 - 1) * self.limits

    def _spherical2screen(self, coord_spherical):
        """ In range: [-FoV_lens/2, FoV_lens/2], out range: [0, 1] """
        x, y = coord_spherical.T
        horiz_limit, vert_limit = self.limits
        x = (x / horiz_limit + 1.) * 0.5
        y = (y / vert_limit + 1.) * 0.5
        return np.array([x, y], dtype=DT_FLOAT).T

    def screen2ptz(self, x, y, f=None):
        pts_screen = np.array(
            [x, y], dtype=DT_FLOAT
        ) / self.frame_orig_size

        pts_spherical = self._screen2spherical(pts_screen)
        pan_deg, tilt_deg = np.rad2deg(
            self._gnomonic_inverse(pts_spherical, (0, 0))
        )
        f = f if f is not None else self.zoom_f
        return pan_deg, tilt_deg, f

    def ptz2screen(self, pan_deg, tilt_deg, f=None):
        pts_spherical = np.deg2rad(
            np.array([pan_deg, tilt_deg], dtype=DT_FLOAT)
        )

        pts_spherical = self._gnomonic(pts_spherical, (0, 0))
        pts_screen = self._spherical2screen(pts_spherical)
        return (pts_screen * self.frame_orig_size).astype(DT_INT)

    def _spherical2screen_fov(
        self,
        pts_spherical,
        normalized=False
    ):
        # Scale down the coords to match the current FoV
        pts_spherical_fov = pts_spherical * \
            (self.fov_rad / 2 / self.limits)

        # Map to sphere, compensate pitch tilt
        # pts_spherical_fov = self._gnomonic_inverse(
        #     pts_spherical_fov, (0, 0)
        # )
        pts_spherical_fov = self._gnomonic_inverse(
            pts_spherical_fov, (0, self.pitch_tilt_deg)
        )

        # # Project to plane, use target point as gnomonic center
        pts_spherical_fov = self._gnomonic(pts_spherical_fov)

        # Convert to screen coordinates
        pts_screen_fov = self._spherical2screen(pts_spherical_fov)

        if normalized:
            return pts_screen_fov

        return (pts_screen_fov * self.frame_orig_size).astype(DT_INT)

    # region Frame points
    @staticmethod
    def _get_pts_frame_screen():
        xx, yy = np.meshgrid(np.linspace(0, 1, Camera.FRAME_W),
                             np.linspace(0, 1, Camera.FRAME_H))
        return np.array([xx.ravel(), yy.ravel()], dtype=DT_FLOAT).T

    @cached_property
    def _pts_frame_spherical(self):
        pts_screen_frame = RectilinearCamera._get_pts_frame_screen()
        return self._screen2spherical(pts_screen_frame)

    def __get_frame_roi(self, frame_orig):
        """Get ROI by sampling all pixels from the sphere.

        DEPRECATED: Using homography in self.get_frame_roi instead.
        """
        pts_fov_screen = self._spherical2screen_fov(
            self._pts_frame_spherical,
            normalized=True
        )
        return self._remap(frame_orig, pts_fov_screen)

    def _remap(self, frame_orig, pts_screen):
        """Creates a new frame by mapping from frame_orig based on pts.

        DEPRECATED: Homography used instead.

        Args:
            frame_orig: Frame to be sampled from.
            pts: Mapping scheme.
                Range: [0, 1]

        Returns:
            Remapped image
                Range: [0, frame_size]
        """
        frame_orig_h, frame_orig_w, _ = frame_orig.shape

        map_x = pts_screen[:, 0] * frame_orig_w
        map_y = pts_screen[:, 1] * frame_orig_h
        map_x = np.reshape(map_x, self.frame_roi_size[::-1])
        map_y = np.reshape(map_y, self.frame_roi_size[::-1])

        return cv2.remap(frame_orig, map_x, map_y, interpolation=INTERPOLATION_TYPE)
    # endregion

    # region Corner points
    @staticmethod
    def _get_pts_corners_screen():
        return np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ], dtype=DT_FLOAT)

    @cached_property
    def _pts_corners_spherical(self):
        pts_screen_frame = RectilinearCamera._get_pts_corners_screen()
        return self._screen2spherical(pts_screen_frame)

    def get_pts_corners(self):
        return self._spherical2screen_fov(self._pts_corners_spherical)

    # endregion

    # region Border points

    @staticmethod
    def _get_pts_borders_screen():
        DRAWING_STEP = 25

        w = Camera.FRAME_W // DRAWING_STEP
        h = Camera.FRAME_H // DRAWING_STEP
        pts_w = np.linspace(0, 1, w)
        pts_h = np.linspace(0, 1, h)

        top = np.array([
            pts_w.copy(),
            np.zeros(w)
        ]).T
        right = np.array([
            np.ones(h),
            pts_h.copy()
        ]).T
        bottom = np.array([
            np.flip(pts_w.copy()),
            np.ones(w)
        ]).T
        left = np.array([
            np.zeros(h),
            pts_h.copy()
        ]).T

        return np.concatenate([top, right, bottom, left])

    @cached_property
    def _pts_borders_spherical(self):
        pts_borders_screen = RectilinearCamera._get_pts_borders_screen()
        return self._screen2spherical(pts_borders_screen)

    def get_pts_borders(self):
        """Get points along the border of the current ROI in frame space."""
        return self._spherical2screen_fov(
            self._pts_borders_spherical
        )
    # endregion

    def _gnomonic(self, coord_spherical, center_deg=None):
        """ Convert latitude (tilt) and longtitude (pan) to x, y coordinates.

        The conversion is based on the Gnomonic Projection.
        https://mathworld.wolfram.com/GnomonicProjection.html

        Args:
            coord_spherical:
                Range: [-FoV_lens/2, FoV_lens/2]
            center_deg: Center of projection in degrees.

        Returns:
            x, y coordinates
                Range: [-FoV_lens/2, FoV_lens/2]
        """
        lambda_rad = coord_spherical.T[0]  # pan
        phi_rad = coord_spherical.T[1]  # tilt

        if center_deg is None:
            center_deg = np.array(
                [self.pan_deg, self.tilt_deg + self.pitch_tilt_deg]
            )
        center_pan_rad, center_tilt_rad = -np.deg2rad(center_deg)

        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)
        cos_c = np.sin(center_tilt_rad) * sin_phi + np.cos(center_tilt_rad) * \
            cos_phi * np.cos(lambda_rad - center_pan_rad)
        x = (cos_phi * np.sin(lambda_rad - center_pan_rad)) / cos_c
        y = (np.cos(center_tilt_rad) * sin_phi - np.sin(center_tilt_rad) *
             cos_phi * np.cos(lambda_rad - center_pan_rad)) / cos_c

        return np.array([x, y]).T

    def _gnomonic_inverse(self, coord_spherical, center_deg=None):
        """Convert x, y coordinates to latitude (tilt) and longtitude (pan).

        It assumes that the x, y coordinates have been obtained by the Gnomonic Projection.
        https://mathworld.wolfram.com/GnomonicProjection.html

        Args:
            coord_spherical:
                In range: [-FoV_lens/2, FoV_lens/2]
                Out range: [-FoV_lens/2, FoV_lens/2]
            center: Center of projection in degrees.
        """
        x = coord_spherical.T[0]
        y = coord_spherical.T[1]

        if center_deg is None:
            center_deg = np.array(
                [self.pan_deg, self.tilt_deg + self.pitch_tilt_deg]
            )
        center_pan_rad, center_tilt_rad = -np.deg2rad(center_deg)

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(
            cos_c * np.sin(center_tilt_rad) + (y * sin_c * np.cos(center_tilt_rad)) / rou)
        lon = center_pan_rad + np.arctan2(x * sin_c, rou * np.cos(center_tilt_rad)
                                          * cos_c - y * np.sin(center_tilt_rad) * sin_c)

        return np.array([lon, lat]).T

    # -------------------------------------------------------------------------
    # Drawing

    def draw_roi_(self, frame_orig, color=Color.VIOLET, drawing_mode=DrawingMode.LINES):
        THICKNESS_BIG = 40
        THICKNESS_SMALL = 10

        pts = self.get_pts_borders()
        if drawing_mode == DrawingMode.LINES:
            cv2.polylines(frame_orig, [pts], True,
                          color, thickness=THICKNESS_BIG)
        elif drawing_mode == DrawingMode.CIRCLES:
            for x, y in pts:
                cv2.circle(frame_orig, [x, y], radius=5,
                           color=color, thickness=-1)

    def draw_frame_mask(self, frame_orig):
        mask = np.zeros(frame_orig.shape[:2], dtype=np.uint8)
        pts = self.get_pts_borders()
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(frame_orig, frame_orig, mask=mask)

    def draw_grid_(self, frame_orig, color=Color.BLUE, drawing_mode=DrawingMode.LINES):
        THICKNESS = 8
        RANGE = np.pi/2 if drawing_mode == DrawingMode.LINES else np.pi/2
        STEPS = 20

        pan_rad = np.deg2rad(self.pan_deg)
        tilt_rad = np.deg2rad(self.tilt_deg)

        xx, yy = np.meshgrid(np.linspace(-RANGE, RANGE, STEPS),
                             np.linspace(-RANGE, RANGE, STEPS))
        pts = np.array([xx.ravel() - pan_rad, yy.ravel() -
                       tilt_rad], dtype=DT_FLOAT).T

        # pts = self._gnomonic(pts, (0, 0))
        pts = self._gnomonic_inverse(pts, (0, self.pitch_tilt_deg))
        pts = self._gnomonic(pts)

        pts = self._spherical2screen(pts)
        pts = (pts * self.frame_orig_size).astype(DT_INT)

        if drawing_mode == DrawingMode.LINES:
            xx, yy = pts.T
            xx, yy = xx.reshape((STEPS, STEPS)), yy.reshape((STEPS, STEPS))

            for i in range(STEPS):
                pts = np.array([xx[i].ravel(), yy[i].ravel()], dtype=DT_INT).T
                cv2.polylines(frame_orig, [pts], False,
                              color, thickness=THICKNESS)

            for i in range(STEPS):
                pts = np.array(
                    [xx[:, i].ravel(), yy[:, i].ravel()], dtype=DT_INT).T
                cv2.polylines(frame_orig, [pts], False,
                              color, thickness=THICKNESS)
        elif drawing_mode == DrawingMode.CIRCLES:
            for x, y in pts:
                cv2.circle(frame_orig, [x, y], radius=THICKNESS,
                           color=color, thickness=-1)

    def process_input(self, key, mouse_pos):
        is_alive = True
        if key == ord('8'):
            self.sensor_w += RectilinearCamera.SENSOR_DX
        elif key == ord('2'):
            self.sensor_w -= RectilinearCamera.SENSOR_DX
        elif key == ord('6'):
            self.lens_fov_horiz_deg += RectilinearCamera.FOV_DX
        elif key == ord('4'):
            self.lens_fov_horiz_deg -= RectilinearCamera.FOV_DX
        elif key == ord('9'):
            self.pitch_tilt_deg += RectilinearCamera.TILT_DX
        elif key == ord('7'):
            self.pitch_tilt_deg -= RectilinearCamera.TILT_DX
        else:
            is_alive = super().process_input(key, mouse_pos)
        return is_alive

    def get_stats(self):
        stats = {
            "Name": RectilinearCamera.__name__,
            "f": f"{self.zoom_f:.2f}",
            "sensor_w": self.sensor_w,
            "lens_fov_horiz_deg": self.lens_fov_horiz_deg,
            "lens_fov_vert_deg": self.lens_fov_vert_deg,
            "pan_deg": f"{self.pan_deg:.4f}",
            "tilt_deg": f"{self.tilt_deg:.4f}",
            "fov_horiz_deg": self.fov_horiz_deg,
            "fov_vert_deg": self.fov_vert_deg,
            "center": self.center,
            "pitch_tilt_deg": self.pitch_tilt_deg
        }
        return stats
