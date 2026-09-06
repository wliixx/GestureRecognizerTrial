import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = 'wliixx/opencv/hand_landmarker (1).task'
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # большой палец
    (0, 5), (5, 6), (6, 7), (7, 8),          # указательный
    (5, 9), (9, 10), (10, 11), (11, 12),     # средний
    (9, 13), (13, 14), (14, 15), (15, 16),   # безымянный
    (13, 17), (17, 18), (18, 19), (19, 20),  # мизинец
    (0, 17),                                  # ладонь
]

base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options = base_options,
    num_hands = 2,
    min_hand_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
    running_mode = mp_vision.RunningMode.VIDEO
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

TIP_IDS = [4, 8, 12, 16, 20]
