import numpy as np

__STATE_SIZE = (110, 128)
""" State size represented as h and w """

__STATE_SIZE_REV = tuple(reversed(__STATE_SIZE))
""" State size represented as w and h """


def get_state_size() -> tuple:
    return __STATE_SIZE


def preprocess_state(state: np.ndarray):
    import cv2 as cv

    grayscale = cv.cvtColor(state, cv.COLOR_RGB2GRAY)
    rescaled = cv.resize(grayscale, __STATE_SIZE_REV)

    return rescaled


def preprocess_info(info: dict) -> np.array:
    status = info["status"]
    status_numeric = 0
    if status == "small":
        status_numeric = 0.25
    elif status == "tall":
        status_numeric = 0.5
    elif status == "fireball":
        status_numeric = 0.75

    info_out = [float(info["x_pos"]) / 256.0, float(info["y_pos"]) / 240.0, status_numeric]
    return np.array(info_out, dtype=np.float32)
