import cv2

for index in range(5):  # Check indices 0-4
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera found at index {index}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {index}", frame)
            cv2.waitKey(1000)  # Display frame for 1 second
        cap.release()
    else:
        print(f"No camera at index {index}")

cv2.destroyAllWindows()
