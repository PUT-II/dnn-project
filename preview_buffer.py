import os
import shutil

import cv2 as cv
import numpy as np

from udrl.replay_buffer import ReplayBuffer, EpisodeTuple


def preview_buffer():
    if not os.path.isfile('buffer_latest.npy'):
        return

    buffer = ReplayBuffer()
    buffer.load('buffer_latest.npy')

    if os.path.isdir("./preview"):
        shutil.rmtree("./preview")

    os.mkdir("./preview")

    episode: EpisodeTuple
    for i, episode in enumerate(buffer):
        imgs = episode.states
        file_name = f"./preview/{episode.total_return}_{i}.mp4"
        print(f"Saving: {file_name}")

        img_shape = imgs[0].shape
        is_color = len(img_shape) > 2

        if is_color:
            shape = (img_shape[2], img_shape[1])
        else:
            shape = (img_shape[1], img_shape[0])
        fourcc = cv.VideoWriter_fourcc(*"mp4v")

        fps = 30.0
        out = cv.VideoWriter(file_name, fourcc, fps, shape, is_color)
        for img in imgs:
            frame = img
            if is_color:
                frame = np.transpose(frame, (1, 2, 0))
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            out.write(frame)
        out.release()


if __name__ == "__main__":
    preview_buffer()
