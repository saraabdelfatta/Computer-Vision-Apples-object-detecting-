# Object Detecting

A small computer-vision project for detecting and counting objects in video
using Python, OpenCV, and Ultralytics YOLO. The scripts process video frames,
draw detections, and count objects as they cross a configured line.

This repository is intended as a learning project and starting point for
experiments with different object classes, models, and video inputs. Detection
quality depends on the selected model and the footage.

## What the project does

- Reads `video1.mp4` frame by frame.
- Runs Ultralytics' `ObjectCounter` solution with `yolo11n.pt`.
- Selects COCO class ID `39` (bottle) and counts line crossings.
- Displays the annotated frames and writes them to `result.avi`.

The scripts currently use a pretrained general-purpose YOLO model; despite
the original repository name, the configured class is **bottle**, not apple.
`best-detect.pt` is included but is not loaded by the current scripts. Check a
model's class names before changing the selected class ID.

## Files

| File | Description |
| --- | --- |
| `counting.py` | Video object-counting example. |
| `main0402.py` | A second video-processing example with the same current settings. |
| `yolo11n.pt` | YOLO model used by the scripts. |
| `best-detect.pt` | Additional model weights; not used by the scripts by default. |
| `video1.mp4` | Input video used by the scripts. |
| `video.mp4` | Additional video asset. |
| `result.avi` | Example annotated output. |

## Requirements

- Python
- OpenCV
- Ultralytics

Install the Python packages with:

```bash
python -m pip install opencv-python ultralytics
```

## Run

Run either script from the project directory:

```bash
python counting.py
```

Or:

```bash
python main0402.py
```

The script opens a display window and saves the processed video as
`result.avi` in the current directory. Keep the model and input video files
available at the paths expected by the script.

## Configuration

In either script, update the `ObjectCounter` settings to experiment:

- Set `model` to the desired compatible model weights.
- Set `classes` to the class ID or IDs supported by that model.
- Adjust `region` to place the counting line for the input video's resolution.
- Replace `video1.mp4` to use another input video.
- Change `result.avi` to choose another output path.

The input video must open successfully for the script to produce meaningful
output. The counting line and video dimensions may need adjustment for other
footage.

## Next steps

- Confirm and document each model's class labels.
- Add command-line options for input video, model, and output paths.
- Evaluate detections on representative footage and document errors.
- Experiment with other object classes and suitable trained models.
