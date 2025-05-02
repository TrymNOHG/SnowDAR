from ultralytics import YOLO
import cv2


def save_result_img(results, img, rectangle_thickness=2, text_thickness=1):
    for result in results:
       for box in result.boxes:
           cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
           cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                       (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    cv2.imwrite("./result.png", img)


# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")
# change rgb to lidar for training the lidar model
results = model.train(cfg="rgb_train.yaml", epochs=100, lr0=0.005)

# Evaluate the model's performance on the validation set
# results = model.val()

# test_path = "./Poles/rgb/images/test/frame_000005.PNG"
# Perform object detection on an image using the model
# results = model(test_path)
# img = cv2.imread(test_path)
# save_result_img(results, img)

# Export the model to ONNX format
# success = model.export(format="onnx")
