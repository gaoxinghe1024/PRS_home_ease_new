# Data Card and Evaluation Limits

## UI-PRMD raw skeletons

| Item | Record |
|---|---|
| Official source | https://webpages.uidaho.edu/ui-prmd/ |
| Download source | raw text mirror jkaethee/UI-PRMD_Exercise_Model |
| Pinned commit | 475c66aba07731adbe1c90a15462fd30be564593 |
| Data files | 2,000 Positions / Angles text files, 98,438,042 bytes |
| Skeleton segments | 1,000, 22 joints per segment, 30 Hz |
| Participants | 10; not a representative sample of older adults |
| Four target exercises | m05 sit-to-stand, m01 deep squat, m06 straight-leg raise, m02 hurdle step |
| Negative exercise | m07 shoulder raise |
| Correct / incorrect executions | 10 segments each per exercise and person, split by person |
| Train / validation / test | s01-06 / s07-08 / s09-10, 600 / 200 / 200 segments |
| Source ground truth | exercise class, participant, segment repetition number, correct / incorrect execution |
| Weak labels of this project | four-phase labels derived from the kinematic signal of the whole segment |
| Derived input of this project | 25 / 45 / 65 degree camera-plane projections of the real 3D skeletons |
| Inference confidence | projected source skeletons have confidence=1, not the visibility confidence of a real pose CNN |
| Count evaluation | correct segments of two people per exercise concatenated; sit-to-stand 9 per person, the others 10; 78 cycles in total |
| Negative replays | the m07 segments of the two people, which must not be counted |
| Video evidence | no count acceptance on real RGB camera video |

The source URL, SHA-256, Git blob SHA and size of every file are in `data/raw/uiprmd/manifest.json`. The download script pins the commit by default and checks the original blob SHA, so a corrupted cached file of the same size cannot go unnoticed.

"Incorrect execution" in the source is not a general definition of an invalid count: a movement with non-standard range or posture can still complete a cycle. This project has no quality-assessment model. These segments therefore add diversity to action / phase training, while the count reference is built from the correct-execution source segments only; recognizing incorrect executions is not claimed.

Audit plots of the label curves and the event logs of the test participants verify the source cycles and obvious anomalies. This audit cannot replace independent manual per-frame phase annotation. Metrics of exact cycle timing use weak boundaries, while count=9 / 10 uses the source segment reference; the two must not be confused when interpreting results.

## COCO pose-estimation subset

The official person keypoint annotations of COCO 2017 val, taking in image_id order the first 100 non-crowd images with >= 6 annotated keypoints and area > 4096: 198 people and 2,482 visible keypoints. The selection was fixed before running any pose model and is saved in `data/coco/manifest.json`.

It serves a relative comparison of the general perception accuracy and latency of pre-trained pose CNNs. PCK@0.2bbox is a lenient threshold, and the mean OKS is not the full official AP. The subset contains no protocol designed for home training of older adults and no repetition count labels.

## Use and redistribution

Public accessibility of the mirror does not mean the authors granted a new data licence. Research cites the original UI-PRMD authors and the official source; the raw data, images, the Ultralytics library and the pre-trained weights follow the terms of their respective right holders. This data card accompanies the code, weights, data paths and download instructions when they are handed over; third-party resources are never presented as original work of the project.

## Not covered yet

Natural continuous training, older participants, different chair heights, real clothing / lighting / occlusion, people crossing, separate counts per side, negatives from daily activities, and the safety and medical effectiveness of the exercises. Each needs its own data and evaluation; the current results do not replace them.
