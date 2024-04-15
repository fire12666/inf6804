# TP 3

## Installation

1. Download dependencies with the following command: 

    ```pip install -r requirements.txt```

2. Download the *MOT-Challenge* ground truths (available [here](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip)). The extracted ```data``` folder must be located in the root of this project.

3. Download video on Moodle and place it in ```yolo/frames```.

## Project Structure

Project structure should look like this.

```
data
    gt
        ...
        mot_challenge
        ...
    trackers
        ...
        mot_challenge
            ...
            MOT20-train
                MPNTrack <-- Model name
                    data <-- Results go here
                        MOT20-02.txt
                        ...
            ...
        ...
hota_metrics
MOT20
scripts
yolo
    frames
```

## Usage

Following commands must be executed in the root folder.

### Inference

Predict on MOT20 dataset.

> python yolo/yolo_tracking.py

Predict on cups video.

> python yolo/predict_cups.py

### Evaluate on MOT20 Dataset

Replace ```MPNTrack``` with model name.

> python scripts/run_mot_challenge.py --BENCHMARK MOT20 --USE_PARALLEL False --METRICS HOTA --TRACKERS_TO_EVAL MPNTrack

*Results are saved in ```data/trackers/mot_challenge/MOT20-train/MPNTrack/pedestrian_summary.txt```.*
