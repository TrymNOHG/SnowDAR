from PIL import Image
import os

def update_annotation(file_path, y_percent):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip("\n")
        vals = line.split(" ")
        if len(vals) != 6:
            continue
        else:
            x, y, w, h = vals[1:-1]
            new_line = vals[0] + " "
            new_line += x + " "
            new_y =  float(y) * (1 - y_percent) + y_percent
            new_line += str(new_y) + " "
            new_line += w + " " + h + "\n"
            new_lines.append(new_line)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def post_process_annotations(dir, y_percent):
    for file in os.listdir(dir):
        update_annotation(f"{dir}/{file}", y_percent)


if __name__ == "__main__":
    post_process_annotations("./save/predict4/labels", 0.35)