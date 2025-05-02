"""
- Do the data points correspond? For each RGB image there is a corresponsing LiDAR image?
    - NO

- On the images:
    - Resolution.
    LiDAR: 1024x128 1024x128+0+0 16-bit sRGB
    RGB: 1920x1208 1920x1208+0+0 8-bit sRGB
        Quite noisy
"""

"""
LiDAR data point counts:
  train: 1367
  test: 197
  valid: 390

RGB data point counts:
  train: 322
  test: 46
  valid: 92
"""


"""
Average pole stats per image:

LiDAR Train:
  Avg poles/image: 2.01
  Avg normalized position (x, y): (0.501, 0.788)
  Avg normalized size (w, h): (0.007, 0.232)

LiDAR Valid:
  Avg poles/image: 2.02
  Avg normalized position (x, y): (0.501, 0.798)
  Avg normalized size (w, h): (0.007, 0.237)

RGB Train:
  Avg poles/image: 1.22
  Avg normalized position (x, y): (0.461, 0.607)
  Avg normalized size (w, h): (0.009, 0.122)

RGB Valid:
  Avg poles/image: 1.23
  Avg normalized position (x, y): (0.456, 0.605)
  Avg normalized size (w, h): (0.008, 0.112)
"""


import os
import matplotlib.pyplot as plt


lidar_root = "Poles/lidar/"
lidar_data = "combined_color"
rgb_root = "Poles/rgb/"
rgb_data = "images"

splits = ["train", "test", "valid"]

def count_files(base_path, data_folder):
    counts = {}
    for split in splits:
        dir_path = os.path.join(base_path, data_folder, split)
        if os.path.exists(dir_path):
            counts[split] = len([
                f for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
            ])
        else:
            counts[split] = 0
    return counts


def count_dataset_samples():
    lidar_counts = count_files(lidar_root, lidar_data)
    rgb_counts = count_files(rgb_root, rgb_data)
    # lidar_counts = count_files(lidar_root, "labels")
    # rgb_counts = count_files(rgb_root, "labels")

    print("LiDAR data point counts:")
    for split in splits:
        print(f"  {split}: {lidar_counts[split]}")

    print("\nRGB data point counts:")
    for split in splits:
        print(f"  {split}: {rgb_counts[split]}")


def parse_label_file(path):
    poles = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # Ignore class (parts[0])
                x, y, w, h = map(float, parts[1:])
                poles.append((x, y, w, h))
    return poles

def compute_stats(label_dir):
    total_poles = 0
    total_images = 0
    sum_x = sum_y = sum_w = sum_h = 0.0

    if not os.path.exists(label_dir):
        return 0, 0, 0, 0, 0

    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)
        if os.path.isfile(path):
            poles = parse_label_file(path)
            total_poles += len(poles)
            total_images += 1
            for x, y, w, h in poles:
                sum_x += x
                sum_y += y
                sum_w += w
                sum_h += h

    if total_images == 0:
        return 0, 0, 0, 0, 0

    avg_poles = total_poles / total_images
    avg_x = sum_x / total_poles if total_poles else 0
    avg_y = sum_y / total_poles if total_poles else 0
    avg_w = sum_w / total_poles if total_poles else 0
    avg_h = sum_h / total_poles if total_poles else 0

    return avg_poles, avg_x, avg_y, avg_w, avg_h


def count_avg_poles_per_image():
    print("Average pole stats per image:")

    for dataset, root in [("LiDAR Train", lidar_root), ("LiDAR Valid", lidar_root),
                          ("RGB Train", rgb_root), ("RGB Valid", rgb_root)]:
        split = "train" if "Train" in dataset else "valid"
        label_dir = os.path.join(root, "labels", split)
        avg_poles, avg_x, avg_y, avg_w, avg_h = compute_stats(label_dir)

        print(f"\n{dataset}:")
        print(f"  Avg poles/image: {avg_poles:.2f}")
        print(f"  Avg normalized position (x, y): ({avg_x:.3f}, {avg_y:.3f})")
        print(f"  Avg normalized size (w, h): ({avg_w:.3f}, {avg_h:.3f})")


def collect_sizes(label_dir):
    widths = []
    heights = []
    if not os.path.exists(label_dir):
        return widths, heights

    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)
        if os.path.isfile(path):
            poles = parse_label_file(path)
            for _, _, w, h in poles:
                widths.append(w)
                heights.append(h)
    return widths, heights


def plot_pole_size_stats():
    datasets = [
        ("LiDAR Train", os.path.join(lidar_root, "labels/train")),
        ("LiDAR Valid", os.path.join(lidar_root, "labels/valid")),
        ("RGB Train", os.path.join(rgb_root, "labels/train")),
        ("RGB Valid", os.path.join(rgb_root, "labels/valid")),
    ]

    for name, path in datasets:
        widths, heights = collect_sizes(path)
        if not widths or not heights:
            continue

        plt.figure()
        plt.hist(widths, bins=30, alpha=0.7, label='Width')
        plt.hist(heights, bins=30, alpha=0.7, label='Height')
        plt.title(f"{name} - Pole Width/Height Histogram")
        plt.xlabel("Normalized Size")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.scatter(widths, heights, alpha=0.5)
        plt.title(f"{name} - Width vs Height")
        plt.xlabel("Width")
     

def plot_pole_centers():
    image_width = 1920
    image_height = 1208

    datasets = [
        ("RGB Train", os.path.join(rgb_root, "labels/train")),
        ("RGB Valid", os.path.join(rgb_root, "labels/valid")),
    ]

    for name, label_dir in datasets:
        centers_x = []
        centers_y = []

        if not os.path.exists(label_dir):
            continue

        for file in os.listdir(label_dir):
            path = os.path.join(label_dir, file)
            if os.path.isfile(path):
                poles = parse_label_file(path)
                for x, y, _, _ in poles:
                    centers_x.append(x * image_width)
                    centers_y.append(y * image_height)

        if not centers_x or not centers_y:
            continue

        plt.figure(figsize=(10, 6))
        plt.scatter(centers_x, centers_y, alpha=0.5, s=10)
        plt.title(f"{name} - Pole Center Distribution (Scaled to 1920x1208)")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.grid(True)
        plt.xlim(0, image_width)
        plt.ylim(0, image_height)

    plt.gca().invert_yaxis()
    plt.savefig("a.png")


def plot_pole_centers_with_heights():
    image_width = 1920
    image_height = 1208

    datasets = [
        ("RGB Train", os.path.join(rgb_root, "labels/train")),
        ("RGB Valid", os.path.join(rgb_root, "labels/valid")),
    ]

    for name, label_dir in datasets:
        if not os.path.exists(label_dir):
            continue

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        for file in os.listdir(label_dir):
            path = os.path.join(label_dir, file)
            if os.path.isfile(path):
                poles = parse_label_file(path)
                for x, y, _, h in poles:
                    x_pix = x * image_width
                    y_pix = y * image_height
                    h_pix = h * image_height
                    y_top = y_pix - h_pix / 2
                    y_bot = y_pix + h_pix / 2
                    ax.plot([x_pix, x_pix], [y_top, y_bot], color='blue', alpha=0.5, linewidth=1)

        plt.title(f"{name} - Pole Centers + Heights")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.grid(True)
        plt.xlim(0, image_width)
        plt.ylim(0, image_height)
        ax.invert_yaxis()  # Top-left origin like image
        filename = name.lower().replace(" ", "_") + "_centers_with_heights.png"
        plt.savefig(filename, dpi=300)
        plt.close()


def plot_pole_centers_with_height_and_width(modality="rgb"):
    if modality == "rgb":
        image_width = 1920
        image_height = 1208
        root = rgb_root
    elif modality == "lidar":
        image_width = 1024
        image_height = 128
        root = lidar_root
    else:
        raise ValueError("Modality must be either 'rgb' or 'lidar'")

    datasets = [
        (f"{modality.upper()} Train", os.path.join(root, "labels/train")),
        (f"{modality.upper()} Valid", os.path.join(root, "labels/valid")),
    ]

    for name, label_dir in datasets:
        if not os.path.exists(label_dir):
            continue

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.axvline(x=image_width / 2, color='green', linestyle='--', linewidth=1, label='split')

        for file in os.listdir(label_dir):
            path = os.path.join(label_dir, file)
            if os.path.isfile(path):
                poles = parse_label_file(path)
                for x, y, w, h in poles:
                    x_pix = x * image_width
                    y_pix = y * image_height
                    h_pix = h * image_height
                    w_pix = w * image_width

                    # Vertical bar for height
                    y_top = y_pix - h_pix / 2
                    y_bot = y_pix + h_pix / 2
                    ax.plot([x_pix, x_pix], [y_top, y_bot], color='blue', alpha=0.5, linewidth=1)

                    # Horizontal bar for width
                    x_left = x_pix - w_pix / 2
                    x_right = x_pix + w_pix / 2
                    ax.plot([x_left, x_right], [y_pix, y_pix], color='red', alpha=0.5, linewidth=1)

        plt.title(f"{name} - Pole Centers + Height/Width Bars")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.grid(True)
        plt.xlim(0, image_width)
        plt.ylim(0, image_height)
        ax.invert_yaxis()

        filename = name.lower().replace(" ", "_") + "_centers_with_bars.png"
        plt.savefig(filename, dpi=300)
        plt.close()


#count_avg_poles_per_image()
#plot_pole_centers()
plot_pole_centers_with_height_and_width(modality="lidar")

