from ultralytics import YOLO


stream = 'rtmp://rtmp.klpkw.one/live/test'
video = '../../sample_videos/kianu.mp4'
model = YOLO('yolov8m.pt')

model.predict(source=video, line_thickness=2, conf=0.5, show=True, save=True)
