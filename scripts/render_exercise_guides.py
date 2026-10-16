"""Render chair/squat schematics and original training leg references."""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
EDGES = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7),
         (6, 7), (6, 8), (8, 10), (7, 9), (9, 11)]

# COCO shoulders through ankles, viewed obliquely. Heights are illustrative.
STANDING = [(109, 63), (128, 60), (99, 95), (140, 93), (97, 121), (140, 118),
            (113, 133), (128, 131), (109, 178), (132, 177), (108, 226), (135, 226)]
SITTING = [(95, 103), (114, 100), (99, 133), (129, 130), (120, 155), (150, 151),
           (100, 168), (115, 166), (149, 174), (164, 171), (149, 226), (165, 226)]
SQUATTING = [(109, 99), (128, 96), (137, 112), (153, 108), (161, 96), (177, 92),
             (102, 162), (117, 160), (136, 183), (151, 180), (109, 226), (135, 226)]


def main():
    for exercise, baseline, peak in [
        ("chair_sit_to_stand", SITTING, STANDING),
        ("squat", STANDING, SQUATTING),
    ]:
        canvas = Image.new("RGB", (780, 252), "#f5f8f6")
        draw = ImageDraw.Draw(canvas)
        for n, pose in enumerate([baseline, peak, baseline]):
            ox = n * 260
            q = [(x + ox, y) for x, y in pose]
            draw.line((ox + 20, 231, ox + 240, 231), fill="#d0ddd5", width=2)
            if exercise == "chair_sit_to_stand":
                draw.line([(ox + 77, 118), (ox + 77, 175), (ox + 125, 175)], fill="#9aa8a1", width=4)
                for x in [80, 122]:
                    draw.line((ox + x, 175, ox + x, 228), fill="#9aa8a1", width=4)
            for a, b in EDGES:
                color = "#2f8164" if a % 2 == 0 else "#537eaf"
                draw.line([q[a], q[b]], fill=color, width=5)
            for x, y in q:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="white", outline="#4c7766", width=1)
            hx, hy = (q[0][0] + q[1][0]) / 2, (q[0][1] + q[1][1]) / 2 - 19
            draw.ellipse((hx - 10, hy - 10, hx + 10, hy + 10), outline="#4c7766", width=4)
            draw.text((ox + 14, 10), str(n + 1), fill="#4c7766", font_size=18)
        canvas.save(ROOT / f"web/static/media/{exercise}-guide.png")
        canvas.crop((260, 0, 520, 252)).save(ROOT / f"web/static/media/{exercise}-pose.png")
    from render_training_guides import main as render_training_references
    render_training_references()


if __name__ == "__main__":
    main()
