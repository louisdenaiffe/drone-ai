import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageGrab


print("Loading YOLOv8...")
model = YOLO('yolov8n.pt') 


def run_drone_ai():
    print("AI Started!")
    print("STEP 1: Move the UxPlay window to the TOP-LEFT of your screen.")
    print("STEP 2: Move this AI window to the BOTTOM-RIGHT to avoid feedback.")
    print("Press 'q' to quit.")

    while True:
        screen = ImageGrab.grab(bbox=(0, 0, 1000, 1000))

        frame = np.array(screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame, conf=0.3, verbose=False)

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # Class 0 is 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "HUMAN", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
        resized_frame = cv2.resize(frame, (800, 700))
        cv2.imshow("AI Analytics Feed", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_drone_ai()