from demo import load_checkpoints, make_animation
from skimage import img_as_ubyte
import imageio
import numpy as np
import skimage.transform

# Load image and video
source_image = imageio.imread("source.jpg")
driving_video = imageio.mimread("driving.mp4")

# Load model
generator, kp_detector = load_checkpoints(
    config_path="config/vox-256.yaml",
    checkpoint_path="vox-cpk.pth.tar"
)

# Generate animation
predictions = make_animation(
    skimage.transform.resize(source_image, (256, 256)),
    [skimage.transform.resize(frame, (256, 256)) for frame in driving_video],
    generator,
    kp_detector
)

# Save output
imageio.mimsave(
    "output.mp4",
    [img_as_ubyte(frame) for frame in predictions],
    fps=30
)

print("Animation generated successfully!")
