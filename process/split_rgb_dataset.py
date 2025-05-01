import os
from PIL import Image

# Root of dataset
rgb_root = "Poles/new_rgb/"
splits = ["train", "valid", "test"]

# Input and output subdirectories
image_subdir = "images"
label_subdir = "labels"
output_image_subdir = "split_images"
output_label_subdir = "split_labels"

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

            # Open and split the image
            img = Image.open(img_path)
            w_full, h_full = img.size
            half_width = w_full // 2

            left_img = img.crop((0, 0, half_width, h_full))
            right_img = img.crop((half_width, 0, w_full, h_full))

            # Save split images
            left_img.save(os.path.join(img_output_dir, f"{base}_l.png"))
            right_img.save(os.path.join(img_output_dir, f"{base}_r.png"))

            # Skip labels if missing (e.g., test set)
            if not os.path.exists(lbl_path):
                print(f"Processed {fname} (no labels)")
                continue

            left_labels = []
            right_labels = []

            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, w, h = parts
                    x = float(x)

                    if x < 0.5:
                        # Left half: scale x to new 0-1 range
                        new_x = x * 2
                        left_labels.append(f"{cls} {new_x:.6f} {y} {w} {h}")
                    else:
                        # Right half: shift and scale
                        new_x = (x - 0.5) * 2
                        right_labels.append(f"{cls} {new_x:.6f} {y} {w} {h}")

            # Save split labels
            if left_labels:
                with open(os.path.join(lbl_output_dir, f"{base}_l.txt"), "w") as f:
                    f.write("\n".join(left_labels) + "\n")

            if right_labels:
                with open(os.path.join(lbl_output_dir, f"{base}_r.txt"), "w") as f:
                    f.write("\n".join(right_labels) + "\n")

            print(f"Processed {fname}")

# Run it
if __name__ == "__main__":
    split_rgb_dataset()
