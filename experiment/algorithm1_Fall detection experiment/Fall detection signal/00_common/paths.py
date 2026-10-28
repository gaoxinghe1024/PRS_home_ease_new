# -*- coding: utf-8 -*-
"""Sensor (accelerometer) path -- data path registry.

Data source: the smartphone tri-axial accelerometer channel recorded synchronously
with URFD (University of Rzeszow Fall Detection Dataset); same origin as the video
path so the two streams can be fused later.

Layout:
  data/urfd/csv/{seq}-acc.csv          raw tri-axial acceleration (t_ms, |a|, ax, ay, az)
  data/urfd/csv/{seq}-data.csv         frame alignment table (frame@30fps, t_ms, |a|)
  data/urfd/csv/urfall-cam0-falls.csv  frame-level fall annotations

This project treats the above data as read-only.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

URFD_ACC_DIR = os.path.join(ROOT, "data", "urfd", "csv")


def acc_csv_dir():
    assert os.path.isdir(URFD_ACC_DIR), f"URFD sensor data directory missing: {URFD_ACC_DIR}"
    return URFD_ACC_DIR


def acc_csv(seq):
    return os.path.join(acc_csv_dir(), f"{seq}-acc.csv")


def align_csv(seq):
    """Frame alignment table (frame@30fps <-> t_ms)."""
    return os.path.join(acc_csv_dir(), f"{seq}-data.csv")


def falls_gt_csv():
    return os.path.join(acc_csv_dir(), "urfall-cam0-falls.csv")
