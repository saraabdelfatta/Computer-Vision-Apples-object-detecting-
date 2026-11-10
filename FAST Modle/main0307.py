#this is differnt model like YOLO and SAM, 
# this is FastSAM model which is used for segmentation tasks.
# The code snippet shows how to create a FastSAM model and run inference on an image 
# using the model. The parameters used in the predict function include 
# the source image, device, retina masks, image size, confidence threshold, 
# and intersection over union threshold. 
# The results of the inference are saved to disk.
from ultralytics import FastSAM

# Create a FastSAM model
model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

# Run inference on an image
everything_results = model.predict(
    source="image.png",
    device="cpu",
    # retina_masks=True is used to enable the generation of high-resolution masks for the segmented objects.
    retina_masks=True,
    # imgsz=1024 is used to set the size of the input image for the model. 
    # A larger image size can improve the accuracy of the segmentation results, 
    # but it may also increase the processing time.
    imgsz=1024,
    # conf=0.4 is used to set the confidence threshold for the model's predictions.
    # Predictions with a confidence score below this threshold will be discarded.
    conf=0.4,
    # iou=0.9 is used to set the intersection over union threshold for the model's predictions.
    # Predictions with an IoU score below this threshold will be discarded.
    iou=0.9,
    save=True,
)