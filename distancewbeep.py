import cv2
import numpy as np
import winsound

# Load the YOLOv3 configuration and weights files
model = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Get the names of the output layers
layer_names = model.getLayerNames()

# Get the unconnected output layers
unconnected_layers = model.getUnconnectedOutLayers()

# Convert the unconnected layers to a list of tuples
unconnected_layers = [(layer,) for layer in unconnected_layers]

# Get the names of the unconnected output layers
names = [layer_names[i[0] - 1] for i in unconnected_layers]

# Set the real-world width of the object in meters
object_width = 0.2

# Set the focal length of the camera in pixels
focal_length = 615

# Load the list of class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set up the video capture
cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()

    # Release the video capture

    # Resize the frame to (416, 416)
    resized_frame = cv2.resize(frame, (416, 416))

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    # Set the input for the model
    model.setInput(blob)

    # Forward pass through the model
    outputs = model.forward(names)

    # Process the outputs
    detected_objects = []
    for output in outputs:
        # Process detection here
        for detection in output:
            # Get the class ID and confidence score of the current detection
            class_id = np.argmax(detection[5:])
            confidence = detection[5 + class_id]

            if confidence > 0.5:
                # Get the bounding box dimensions
                box = detection[:4] * np.array(
                    [resized_frame.shape[1], resized_frame.shape[0], resized_frame.shape[1], resized_frame.shape[0]])
                (center_x, center_y, width, height) = box.astype('int')
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                # Get the name of the detected object
                class_name = classes[class_id]
                detected_objects.append(class_name)

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + int(width), y + int(height)), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Estimate the distance of the object from the camera
                distance = (object_width * focal_length) / width
                cv2.putText(frame, f"{distance:.1f} m", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if distance <= 1:
                    freq = 100
                    dur = 50
                    for i in range(0, 5):
                        winsound.Beep(freq, dur)
                        freq += 100
                        dur += 50
                # time.sleep(5)

                cv2.imshow("frame", frame)

                # quit the program if you press 'q' on keyboard
                if cv2.waitKey(1) == ord("q"):
                    break

cap.release()
cv2.destroyAllWindows()

