# TP 3

## Installation

1. Download dependencies with the following command: 

    ```pip install -r requirements.txt```

2. Download the *MOT-Challenge* ground truths (available [here](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip)). The extracted ```data``` folder must be located in the root of this project.

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
            MOT17-train
                MPNTrack <-- Model name
                    data <-- Results go here
                        MOT17-02-DPM.txt
                        MOT17-02-FRCNN.txt
                        ...
            ...
        ...
hota_metrics
MOT17
scripts
yolo
```

## Usage

Following commands must be executed in the root folder.

### Inference

> python yolo/yolo_tracking.py

### Evaluate on MOT17 Dataset

*Replace ```MPNTrack``` with model name.*

> python scripts/run_mot_challenge.py --USE_PARALLEL False --METRICS HOTA --TRACKERS_TO_EVAL MPNTrack

*Results are saved in ```data/trackers/mot_challenge/MOT17-train/MPNTrack/pedestrian_summary.txt```.*

### Evaluate on MOT20 Dataset

*Replace ```MPNTrack``` with model name.*

> python scripts/run_mot_challenge.py --BENCHMARK MOT20 --USE_PARALLEL False --METRICS HOTA --TRACKERS_TO_EVAL MPNTrack

*Results are saved in ```data/trackers/mot_challenge/MOT20-train/MPNTrack/pedestrian_summary.txt```.*
