import os

def traverse_parallel_dirs(images_dir, labels_dir):
    for root, _, files in os.walk(images_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ".png" in ext.lower():
                relative_path = os.path.relpath(os.path.join(root, file), images_dir)
                label_path = os.path.join(labels_dir, os.path.splitext(relative_path)[0] + ".txt")
                image_path = os.path.join(images_dir, relative_path)
                if os.path.exists(label_path):
                    yield image_path, label_path
                else:
                    yield image_path, None
