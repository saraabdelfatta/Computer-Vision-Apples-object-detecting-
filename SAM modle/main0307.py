# in this part we will learn differnt types of models 

# instead of YOLOv11 will use SAM model for segmentation
# it is more for rearch purposes 
# it is not a real time model 

from ultralytics import SAM

# # Load a model
model = SAM("sam2.1_b.pt")

# # Run inference
model.predict(
    source="image.png", save=True)