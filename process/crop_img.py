from PIL import Image
from dir_traverse import traverse_parallel_dirs
import os

def update_annotation(label_path, y_percent, output_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip("\n")
        vals = line.split(" ")
        if len(vals) != 5:
            continue
        else:
            x, y, w, h = vals[1:]
            new_line = vals[0] + " "
            new_line += x + " "
            new_y =  (float(y) - y_percent) / (1-y_percent)
            new_line += str(new_y) + " "
            new_line += w + " " + h + "\n"
            new_lines.append(new_line)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def crop_img(image_path, label_path, y_percent, output_dir):
    im = Image.open(image_path)
    split_name = image_path.split("/")[-2]
    file_name = image_path.split("/")[-1].split(".")[0]
    w, h = im.size 
    
    left = 0
    top = h * y_percent
    right = w
    bottom = h
    
    im2 = im.crop((left, top, right, bottom))
    im2.save(f"{output_dir}/images/{split_name}/{file_name}.png") # Check this

    # Update the values for the corresponding labels.
    if label_path is not None:
        update_annotation(label_path, y_percent, f"{output_dir}/labels/{split_name}/{file_name}.txt")
    
def crop_dir(output_dir, y_percent):
    if not os.path.exists(f"{output_dir}"):
        os.mkdir(f"{output_dir}")
        os.mkdir(f"{output_dir}/images")
        os.mkdir(f"{output_dir}/images/train")
        os.mkdir(f"{output_dir}/images/test")
        os.mkdir(f"{output_dir}/images/valid")
        os.mkdir(f"{output_dir}/labels/")
        os.mkdir(f"{output_dir}/labels/train")
        os.mkdir(f"{output_dir}/labels/valid")
    for image_path, label_path in traverse_parallel_dirs("./Poles/rgb/images", "./Poles/rgb/labels"):
        crop_img(image_path, label_path, y_percent, output_dir)

if __name__ == "__main__":
    crop_dir("./Poles/new_rgb", 0.35)