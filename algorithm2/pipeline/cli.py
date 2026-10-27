"""Command line for the algorithm-2 stages.

    python -m algorithm2 <stage> [options]

Stages follow the order they were run in: download, prepare, pose, train, optimize, calibrate,
finalize, select, robustness, audit, replay, video, smoke, report, optimization-report, openapi,
sources. The training stages overwrite the checkpoints they own, so run them on a copy when the
current numbers have to stay reproducible.
"""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m algorithm2", description=__doc__.splitlines()[0])
    stages = parser.add_subparsers(dest="stage", required=True)

    download = stages.add_parser("download", help="fetch the UI-PRMD clips and record their hashes")
    download.add_argument("--movements", nargs="+", default=["m01", "m02", "m05", "m06", "m07"])
    download.add_argument("--revision")

    stages.add_parser("prepare", help="reconstruct skeletons, write weak phase labels, plot audit figures")

    pose = stages.add_parser("pose", help="compare the pretrained pose CNNs on COCO")
    pose.add_argument("--images", type=int, default=100)

    train = stages.add_parser("train", help="train and compare the eleven first-round models")
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--only", nargs="*")

    optimize = stages.add_parser("optimize", help="train the three second-round candidates")
    optimize.add_argument("--epochs", type=int, default=80)
    optimize.add_argument("--seed", type=int, default=2026)
    optimize.add_argument("--only", nargs="*")

    stages.add_parser("calibrate", help="recalibrate the decoder on full validation probabilities")
    stages.add_parser("finalize", help="run the locked model over the test split")

    select = stages.add_parser("select", help="evaluate every first-round checkpoint and pick the default")
    select.add_argument("--preview", action="store_true")

    stages.add_parser("robustness", help="fixed-model view and noise stress test")
    stages.add_parser("audit", help="negative, static and same-run latency checks on the default model")

    demo = stages.add_parser("replay", help="replay held-out sessions through the SDK and optionally HTTP")
    demo.add_argument("--url", help="server URL, e.g. http://127.0.0.1:8002")

    video = stages.add_parser("video", help="count a video file or camera feed")
    video.add_argument("--source", default="0")
    video.add_argument("--target", type=int, default=10)
    video.add_argument("--exercise", choices=["chair_sit_to_stand", "squat", "standing_leg_raise", "hurdle_step"],
                       default="chair_sit_to_stand")
    video.add_argument("--device")
    video.add_argument("--no-display", action="store_true")
    video.add_argument("--output", default="reports/video_session.json")

    stages.add_parser("smoke", help="run the real pose CNN on a static image and expect zero cycles")
    stages.add_parser("sources", help="verify the downloaded source files against their hashes")
    stages.add_parser("report", help="write the result document, figures and artifact hashes")
    stages.add_parser("optimization-report", help="write the optimization document and training curves")
    stages.add_parser("openapi", help="export docs/openapi.json")

    args = parser.parse_args(argv)
    if args.stage == "download":
        from . import data
        data.download(args.movements, args.revision or data.REVISION)
    elif args.stage == "prepare":
        from . import data
        data.prepare()
    elif args.stage == "pose":
        from . import pose as pose_benchmark
        pose_benchmark.benchmark(args.images)
    elif args.stage == "train":
        from . import train as comparison
        comparison.train(args.epochs, args.seed, args.only)
    elif args.stage in {"optimize", "calibrate", "finalize"}:
        from . import optimize
        records, skeletons = optimize.load()
        if args.stage == "optimize":
            optimize.train(records, skeletons, args.epochs, args.seed, args.only)
        elif args.stage == "calibrate":
            optimize.calibrate(records, skeletons)
        else:
            optimize.finalize(records, skeletons)
    elif args.stage == "select":
        from . import select as selection
        selection.select(args.preview)
    elif args.stage == "robustness":
        from . import audit
        audit.robustness()
    elif args.stage == "audit":
        from . import audit
        audit.fixed_model()
    elif args.stage == "replay":
        from . import replay
        replay.demo(args.url)
    elif args.stage == "video":
        from . import replay
        replay.video(args.source, args.target, args.exercise, args.device, not args.no_display, args.output)
    elif args.stage == "smoke":
        from . import replay
        replay.smoke()
    elif args.stage == "sources":
        from . import audit
        audit.sources()
    elif args.stage == "report":
        from . import report
        report.write_results()
    elif args.stage == "optimization-report":
        from . import report
        report.write_optimization()
    elif args.stage == "openapi":
        from . import webapp
        webapp.export_openapi()
    return 0


if __name__ == "__main__":
    sys.exit(main())
