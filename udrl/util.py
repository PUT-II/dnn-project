import numpy as np

__STATE_SIZE = (110, 128)
""" State size represented as h and w """

__STATE_SIZE_REV = tuple(reversed(__STATE_SIZE))
""" State size represented as w and h """

__USE_RGB = False

__STATE_CHANNELS = 3 if __USE_RGB else 1

__STATUS_DICT = {
    "small": 0.25,
    "tall": 0.5,
    "fireball": 0.75
}


def clip_reward(reward, min_reward, max_reward):
    max_clipped = min(reward, max_reward)
    min_clipped = max(max_clipped, min_reward)
    return min_clipped


def get_state_channels() -> int:
    return __STATE_CHANNELS


def get_state_size() -> tuple:
    return __STATE_SIZE


def preprocess_state(state: np.ndarray):
    import cv2 as cv
    result = state

    if not __USE_RGB:
        result = cv.cvtColor(result, cv.COLOR_RGB2GRAY)
    result = cv.resize(result, __STATE_SIZE_REV)

    if __STATE_CHANNELS > 1:
        result = np.transpose(result, (2, 0, 1))

    return result


def preprocess_info(info: dict) -> np.array:
    status = info["status"]
    status_numeric = 0
    if status in __STATUS_DICT:
        status_numeric = __STATUS_DICT[status]

    info_out = [status_numeric]
    return np.array(info_out, dtype=np.float32)
