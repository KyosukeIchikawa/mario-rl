import cv2
import numpy as np


def make_video(frames: list, path: str):
    """Make video from frames.

    :param frames: list of frames
    :param path: path to save video
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = frames[0].shape
    video = cv2.VideoWriter(path, fourcc, 30, (width, height))
    for frame in frames:
        dst = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(np.array(dst))
    video.release()
