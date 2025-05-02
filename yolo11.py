from ultralytics import YOLO
import cv2

# Create a new YOLO model from scratch
# model = YOLO("yolo11n.yaml")

def save_result_img(results, img, rectangle_thickness=2, text_thickness=1):
    for result in results:
       for box in result.boxes:
           cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
           cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                       (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    cv2.imwrite("./result.png", img)
#    return img, results

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov9e.pt") # n is the nano-version of the YOLOv11 model and this one is specifically for object detection.

# Ultralytics uses a yaml file to specify the location of the datasets. I might need to specify the pre-processing steps in this yaml file.

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(cfg="rgb_train.yaml")

# Evaluate the model's performance on the validation set
# results = model.val()

# test_path = "./Poles/rgb/images/test/frame_000005.PNG"
# # Perform object detection on an image using the model
# results = model(test_path)

# img = cv2.imread(test_path)
# save_result_img(results, img)

# Export the model to ONNX format
# success = model.export(format="onnx")
