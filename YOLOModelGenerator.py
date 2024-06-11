from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(data='data.yaml', epochs=3)

    results = model.val()

    results = model('frame0.jpg')

    success = model.export(format='onnx')