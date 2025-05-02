from PIL import Image
import os

def update_annotation_lines(file_path, y_percent):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    side = file_path.split("/")[-1].split(".")[0][-1]

    new_lines = []
    for line in lines:
        line = line.strip("\n")
        vals = line.split(" ")
        if len(vals) != 6:
            continue
        else:
            x, y, w, h, c = vals[1:]
            x = float(x) / 2 if side == "l" else float(x) / 2 + 0.5
            new_line = vals[0] + " "
            new_line += str(x) + " "
            new_y = float(y) * (1 - y_percent) + y_percent
            new_line += str(new_y) + " "
            h = float(h) * (1 - y_percent)
            new_line += w + " " + str(h) + " " + c + "\n"
            new_lines.append(new_line)
    
    return new_lines

def post_process_annotations(dir, y_percent):
    files = os.listdir(dir)
    processed = set()

    for file in files:
        if not (file.endswith("_l.txt") or file.endswith("_r.txt")):
            continue

        base_name = file[:-6]
        if base_name in processed:
            continue

        output_lines = []
        for suffix in ["_l.txt", "_r.txt"]:
            side_file = base_name + suffix
            full_path = os.path.join(dir, side_file)
            if os.path.exists(full_path):
                output_lines.extend(update_annotation_lines(full_path, y_percent))

        combined_file_path = os.path.join(dir, base_name + ".txt")
        with open(combined_file_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)

        for suffix in ["_l.txt", "_r.txt"]:
            side_file = base_name + suffix
            full_path = os.path.join(dir, side_file)
            if os.path.exists(full_path):
                os.remove(full_path)

        processed.add(base_name)


def update_coco_annotation_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    side = file_path.split("/")[-1].split(".")[0][-1]
    # width, height = (1920, 1208) # rgb
    width, height = (1024, 128) # lidar
    new_lines = []
    for line in lines:
        line = line.strip("\n")
        vals = line.split(" ")
        if len(vals) != 6:
            continue
        else:
            x, y, w, h, c = vals[1:]
            x = float(x)/width
            y = float(y)/height
            w = float(w) / width
            h = float(h) / height
            new_line = vals[0] + " "
            new_line += str(x) + " "
            new_line += str(y) + " "
            new_line += str(w) + " " + str(h) + " " + c + "\n"
            new_lines.append(new_line)

    with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

def post_process_coco_annotations(dir):
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir, file)
        update_coco_annotation_lines(file_path)

if __name__ == "__main__":
    # post_process_annotations("./save/predict5/labels", 0.35)
    post_process_coco_annotations("./lidar_val_predictions")
