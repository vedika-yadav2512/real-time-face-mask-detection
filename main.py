"""
AUTHOR - PINKEY YADAV
TASK - REAL-TIME FACE MASK DETECTION
"""
# importing required libraries
import cv2
import dlib
import imutils.face_utils as face

# Initialising Detector

detector = dlib.get_frontal_face_detector()
# Reading Video
capture = cv2.VideoCapture("./video.mp4")
while True:
    # Reading Video frame by frame
    ret, frame = capture.read()
    # Resizing Frame
    frame = cv2.resize(frame, (1024, 512))
    # Converting to GrayScale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    for rect in rects:
        # Detecting bounding box for faces without mask
        (bX, bY, bW, bH) = face.rect_to_bb(rect)
        # Drawing bounding box
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 0, 255), 3)
        cv2.putText(frame, text="No Mask", org=(bX, bY), fontScale=1,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                    thickness=2)
    # Showing the detected frames of video
    cv2.imshow('Window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
# Closing all opencv windows
cv2.destroyAllWindows()
