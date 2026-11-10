# Apple Object Detection - Computer Vision Practice

This repository is a hands-on project for building practical computer vision
skills and presenting that learning as part of a technical portfolio. It
explores object detection and line-crossing counts in video with Python,
OpenCV, and Ultralytics YOLO.

The goal is to learn by working with real video, model weights, and detection
outputs - not to claim production-ready performance. The project can be extended
with experiments, evaluation results, and notes about what was learned.

## Project files

The scripts, model weights, and video assets are kept on the [`apple` branch](https://github.com/saraabdelfatta/Computer-Vision-Apples-object-detecting-/tree/apple):

- `counting.py` and `main0402.py` - read `video1.mp4`, run Ultralytics'
  `ObjectCounter`, and write the annotated result to `result.avi`.
- `best-detect.pt` and `yolo11n.pt` - model weight files.
- `video.mp4` and `video1.mp4` - video assets; the counting scripts currently
  use `video1.mp4`.
- `result.avi` - an example output video.

## Getting started

Use Python with the packages below installed:

```bash
python -m pip install opencv-python ultralytics
```

Check out the `apple` branch, then run a counting script from the directory
containing the video and model files:

```bash
python counting.py
```

The script opens a display while processing and saves the annotated video as
`result.avi`. Close the display or let the input video finish to end processing.

## Current implementation and next steps

The scripts currently load `yolo11n.pt` and select class index `39`. In the
standard COCO labels used by this model, index `39` is **bottle**, not apple.
For apple detection, use a checkpoint trained for apples (potentially
`best-detect.pt`, after verifying its class names) and set `classes` to the
correct class index for that checkpoint. The scripts do not currently load
`best-detect.pt`.

Good next steps for this learning project:

1. Verify and document the labels in each model checkpoint.
2. Train or fine-tune an apple detector and record the dataset and training
   settings.
3. Evaluate detections on held-out images or video and report suitable metrics.
4. Compare model predictions and counting results, including failure cases.
5. Improve the scripts with configurable input paths, model selection, and
   detection settings.

## Portfolio focus

This project is intended to demonstrate ongoing learning in computer vision:
working with pretrained models, processing video frames, configuring object
detection, and exploring object counting. Add your own experiment notes,
evaluation results, and lessons learned to make the portfolio evidence
reproducible and clearly your own.
