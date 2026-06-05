# BirdDetector 
Ai bird detector for feeding birds



DownloadNow will download several Open Image photos
YOLO_cropper creates croped photos of the downloaded images
Train_crop uses the model to train on the cropped photos
train_wholeframe uses the weights from train_crop to continue the trainging, 'fine tuning', onto the whole_frame photos
predict is used to test the final weights with the model to work on any input image.