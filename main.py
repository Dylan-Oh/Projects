import multiprocessing
from ultralytics import YOLO

if __name__ == '__main__':
    multiprocessing.freeze_support() # windows에서는 필수
    # Load model
    # model = YOLO("yolov8n.pt") # build a new model from scratch
    model = YOLO("yolov8n.pt") # load a pretrained model(recommanded for training)

    # use teh model
    model.train(data=r"C:\Users\User\pyolo8Test\dataset\mask300\data.yaml",
                epochs=30,
                batch=-1,
                imgsz=640) # train the model

    model.val() # evaluate model performance on th validation set
    model(r"C:\Users\User\pyolo8Test\dataset\mask300\images\val\1.jpg") #predict on an image
    seccess = model.export(fromat="onnx") # export the model to onnx format
