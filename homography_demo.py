from pathlib import Path
import cv2
import numpy as np

from soccerpitch import SoccerPitch
from utils import coords_to_pts, load_json

HOMOGRAPHY_VAR = np.array([[0.00039355314220301807, 0.00017112625937443227, 0.24676693975925446], [0.00018934956460725516,
                                                                                                   0.0014098226092755795, -0.6709916591644287], [-2.5851528917542055e-08, 2.747611688391771e-05, 0.007502786349505186]])

HOMOGRAPHY_MAIN = np.array([[0.0006387016037479043, 0.0005930199986323714, -0.22062993049621582], [-7.935698704386596e-06,
                                                                                                   0.0022633650805801153, -0.6155617237091064], [-8.790566994321125e-07, 3.4035128919640556e-05, 0.007194302976131439]])

pitch = SoccerPitch()
pitch_coords = load_json("assets/coords_pitch_model.json")
corners_n = np.array([
    pitch.bottom_left_corner,
    pitch.top_left_corner,
    pitch.top_right_corner,
    pitch.bottom_right_corner,
])
corners_i = coords_to_pts(pitch_coords)
H_n_i, _ = cv2.findHomography(corners_n, corners_i)

WIN_VAR = "var"
WIN_VAR_WARPED = "var_warped"
WIN_MAIN = "main"
WIN_MAIN_WARPED = "main_warped"
WINDOW_FLAGS = cv2.WINDOW_NORMAL
cv2.namedWindow(WIN_VAR, WINDOW_FLAGS)
cv2.namedWindow(WIN_VAR_WARPED, WINDOW_FLAGS)
cv2.namedWindow(WIN_MAIN, WINDOW_FLAGS)
cv2.namedWindow(WIN_MAIN_WARPED, WINDOW_FLAGS)

folder = Path("data_homography")
im_var = cv2.imread(str(folder / "VAR_03.jpg"))
im_main = cv2.imread(str(folder / "Main_03.jpg"))

im_var = cv2.resize(im_var, (960, 540))
im_main = cv2.resize(im_main, (960, 540))

H = H_n_i @ HOMOGRAPHY_VAR
im_var_warped = cv2.warpPerspective(im_var, H, (1574, 1019))
H = H_n_i @ HOMOGRAPHY_MAIN
im_main_warped = cv2.warpPerspective(im_main, H, (1574, 1019))

cv2.imshow(WIN_VAR, im_var)
cv2.imshow(WIN_VAR_WARPED, im_var_warped)
cv2.imshow(WIN_MAIN, im_main)
cv2.imshow(WIN_MAIN_WARPED, im_main_warped)

cv2.waitKey(0)

cv2.destroyAllWindows()
