from PIL import Image
import glob2

files = glob2.glob("kitti_data/**/*.png", recursive=True)

for file in files:
    img = Image.open(file)
    
    img = Image.save(file)
    