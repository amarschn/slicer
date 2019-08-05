from PIL import Image, ImageDraw
import numpy as np

filename = "./test_image.bmp"
im = Image.new("1", (10, 10), 1)
draw = ImageDraw.Draw(im)
rectangle = np.array([[1.,1],[9,1],[9,9],[1,9.]])
rectangle = np.round(rectangle*0.9)
# rectangle -= 1

polygon = tuple(map(tuple, rectangle))

draw.polygon(polygon, fill=0)
im.save(filename)
