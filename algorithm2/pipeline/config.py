"""Values that can change a result: model grids, projections, movements, pose candidates."""

# first round: eleven temporal models over the same labels and subject split
CONFIGS = [
    {"name": "tcn_b2_w32", "kind": "tcn", "width": 32, "blocks": 2},
    {"name": "tcn_b3_w32", "kind": "tcn", "width": 32, "blocks": 3},
    {"name": "tcn_b4_w32", "kind": "tcn", "width": 32, "blocks": 4},
    {"name": "tcn_b3_w64", "kind": "tcn", "width": 64, "blocks": 3},
    {"name": "ds_tcn_b3_w32", "kind": "tcn", "width": 32, "blocks": 3, "depthwise": True},
    {"name": "ds_tcn_b4_w64", "kind": "tcn", "width": 64, "blocks": 4, "depthwise": True},
    {"name": "gru_b2_w32", "kind": "gru", "width": 32, "blocks": 2},
    {"name": "gru_b2_w64", "kind": "gru", "width": 64, "blocks": 2},
    {"name": "transformer_b2_w32", "kind": "transformer", "width": 32, "blocks": 2},
    {"name": "tcn_coords", "kind": "tcn", "width": 32, "blocks": 3, "features": "coords"},
    {"name": "tcn_kinematics", "kind": "tcn", "width": 32, "blocks": 3, "features": "kinematics"},
]

# second round: the three strong families, retrained on stable80 features
OPTIMIZED_CONFIGS = [
    {"name": "opt_tcn_b4_w48", "kind": "tcn", "width": 48, "blocks": 4, "context": 96},
    {"name": "opt_tcn_b5_w48", "kind": "tcn", "width": 48, "blocks": 5, "context": 128},
    {"name": "opt_gru_b2_w64", "kind": "gru", "width": 64, "blocks": 2, "context": 96},
]

# m07 is a shoulder movement and only ever appears as a negative session
MOVEMENTS = ["m05", "m01", "m06", "m02", "m07"]
PROJECTION_ANGLES = [25, 45, 65]

POSE_CANDIDATES = [("yolo11n-pose.pt", 320), ("yolo11n-pose.pt", 640),
                   ("yolo11s-pose.pt", 640), ("yolo11m-pose.pt", 640)]

LABEL_SOURCE = "weak_kinematic_offline_v2"
FPS = 30
