import PIL.Image

from dsine import DSINE
from dsine.utils.visualize import normal_to_rgb

model = DSINE.load_model(
    "dsine/projects/dsine/experiments/exp001_cvpr2024/dsine.txt",
    "dsine/projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt",
    device="cpu",
)

result = model.predict(PIL.Image.open("Profile.jpeg"))

im = PIL.Image.fromarray(normal_to_rgb(result)[0, ...])
im.save("prediction.jpg")
