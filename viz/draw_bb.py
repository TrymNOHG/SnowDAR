import os
from PIL import Image, ImageDraw

# Dataset configuration
rgb_root = "Poles/new_rgb/"
splits = ["train", "valid", "test"]

# Folder structure
image_subdir = "split_images"
label_subdir = "split_labels"
output_subdir = "debug_overlay"

def draw_bbox_on_split_images():
    image_dir = os.path.join(rgb_root, image_subdir)
    label_dir = os.path.join(rgb_root, label_subdir)
    output_dir = os.path.join(rgb_root, output_subdir)

    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        image_split_dir = os.path.join(image_dir, split)
        label_split_dir = os.path.join(label_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)

        for fname in os.listdir(image_split_dir):
            if not fname.endswith(".PNG"):
                continue

            base = os.path.splitext(fname)[0]
            img_path = os.path.join(image_split_dir, fname)
            lbl_path = os.path.join(label_split_dir, base + ".txt")

            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size

            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        _, x, y, w, h = map(float, parts)
                        x_pix = x * img_width
                        y_pix = y * img_height
                        w_pix = w * img_width
                        h_pix = h * img_height

                        x0 = x_pix - w_pix / 2
                        y0 = y_pix - h_pix / 2
                        x1 = x_pix + w_pix / 2
                        y1 = y_pix + h_pix / 2

                        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

            img.save(os.path.join(output_split_dir, fname))

# Run it
if __name__ == "__main__":
    draw_bbox_on_split_images()
