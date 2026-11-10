import cv2
from ultralytics import solutions

#video capture to readthe video file. 
cap = cv2.VideoCapture("video1.mp4")

# we are extracting the weight, height, and FPS of the video file.
# w is the width of the video, 
# h is the height of the video, 
# and fps is the frames per second of the video.
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH,
              cv2.CAP_PROP_FRAME_HEIGHT,
              cv2.CAP_PROP_FPS))

# we are initializing the video writer that it will use to write the output of the solutions.
video_writer = (
    cv2.VideoWriter(
        "result.avi",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)))

# ObjectCounting Module Initialization
solution = solutions.ObjectCounter(
    model="yolo11n.pt",
    classes=[39],
    show_in=True, show_out=True,
    show=True,
    region=[(640, 0), (640, 720)],
    line_width=4,
)

# Process video
# looping through the video frames and processing each frame using the ObjectCounter solution.
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = solution(im0)
    video_writer.write(results.plot_im)
cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy windows
