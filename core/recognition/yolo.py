from ultralytics import YOLO


# any stream, any video, any image can be used for recognition on-the-go
stream = 'rtmp://rtmp.example.com/live/test'
video = '../../sample_videos/kianu.mp4'
model = YOLO('yolov8m.pt')

model.predict(source=video, line_thickness=2, conf=0.5, show=True, save=True)
# YOLO v8 standard version can recognize 80 classes
# should be enough for most of scenes recognition, but can be trained additionally
# or someone's version can be used instead of standard
