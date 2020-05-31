import time

import numpy as np
import cv2

INPUT_STRING = "Cat"
bits = list(str(bin(int.from_bytes(INPUT_STRING.encode(), 'big'))))

ALPHA = 0.5

W = 1920
H = 1280

frame = cv2.imread("Morges Sunset Cropped.png")

BRIGHT = frame.astype(np.uint8)
DULL = (frame * ALPHA).astype(np.uint8)

BRIGHT = np.zeros((H, W, 3)).astype(np.uint8)
DULL = np.zeros((H, W, 3)).astype(np.uint8)


# 12 Frames per bit
def generate_one():
    return [DULL, BRIGHT, DULL, BRIGHT, DULL, BRIGHT, DULL, BRIGHT, DULL, BRIGHT, DULL, BRIGHT]

def generate_zero():
    return [DULL, DULL, BRIGHT, DULL, DULL, BRIGHT, DULL, DULL, BRIGHT, DULL, DULL, BRIGHT]

# 6 Frames per bit
# def generate_one():
#     return [DULL, BRIGHT, DULL, BRIGHT, DULL, BRIGHT]

# def generate_zero():
#     return [DULL, DULL, BRIGHT, DULL, DULL, BRIGHT]

all_frames = []
for bit in bits:
    if bit == "1":
        all_frames += generate_one()
    elif bit == "0":
        all_frames += generate_zero()

# Define the codec and create VideoWriter Object
out = cv2.VideoWriter("{}_12_alpha{}.mp4".format(INPUT_STRING, ALPHA), cv2.VideoWriter_fourcc(*'mp4v'), 30., (1920, 1280))

for frame in all_frames:
    out.write(frame)

out.release()


