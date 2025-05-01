import os
from PIL import Image

rgb_root = "Poles/new_rgb/"
splits = ["train", "valid", "test"]

image_subdir = "images"
label_subdir = "labels"

output_image_subdir = "split_images"
output_label_subdir = "split_labels"

# Image dimensions
image_width = 1920
image_height = 1208
half_width = image_width // 2


def split_rgb_dataset():
    for split in splits:
        img_input_dir = os.path.join(rgb_root, image_subdir, split)
        lbl_input_dir = os.path.join(rgb_root, label_subdir, split)

        img_output_dir = os.path.join(rgb_root, output_image_subdir, split)
        lbl_output_dir = os.path.join(rgb_root, output_label_subdir, split)

        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(lbl_output_dir, exist_ok=True)

        for fname in os.listdir(img_input_dir):
            if not fname.endswith(".png"):
                continue

            base = os.path.splitext(fname)[0]
            img_path = os.path.join(img_input_dir, fname)
            lbl_path = os.path.join(lbl_input_dir, base + ".txt")

            # Open and split image
            img = Image.open(img_path)
            left_img = img.crop((0, 0, half_width, image_height))
            right_img = img.crop((half_width, 0, image_width, image_height))

            # Save split images
            left_name = f"{base}_l.png"
            right_name = f"{base}_r.png"
            left_img.save(os.path.join(img_output_dir, left_name))
            right_img.save(os.path.join(img_output_dir, right_name))

            # Skip test labels since they dont exist
            if not os.path.exists(lbl_path):
                continue

            # Prepare label lists
            left_labels = []
            right_labels = []

            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls, x, y, w, h = parts
                        x = float(x)
                        w = float(w)

                        abs_x = x * image_width

                        if abs_x < half_width:
                            # Goes in left image
                            new_x = abs_x / half_width
                            new_w = w * image_width / half_width
                            left_labels.append(f"{cls} {new_x:.6f} {y} {new_w:.6f} {h}")
                        else:
                            # Goes in right image
                            new_x = (abs_x - half_width) / half_width
                            new_w = w * image_width / half_width
                            right_labels.append(f"{cls} {new_x:.6f} {y} {new_w:.6f} {h}")

            # Save split labels
            with open(os.path.join(lbl_output_dir, f"{base}_l.txt"), 'w') as f:
                if left_labels:
                    f.write("\n".join(left_labels) + "\n")
            with open(os.path.join(lbl_output_dir, f"{base}_r.txt"), 'w') as f:
                if right_labels:
                    f.write("\n".join(right_labels) + "\n")

            print(f"Processed {fname}")


# Run the function
split_rgb_dataset()
