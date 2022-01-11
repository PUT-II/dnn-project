import os
import shutil

import cv2 as cv

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
        file_name = f"./preview/{i}.mp4"
        print(f"Saving: {file_name}")

        shape = (imgs[0].shape[1], imgs[0].shape[0])
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(file_name, fourcc, 20.0, shape, 0)
        for img in imgs:
            out.write(img)
        out.release()


if __name__ == "__main__":
    preview_buffer()
