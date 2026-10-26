"""Render the fourth reviewed version, retaining prior versions."""
import sys
from pathlib import Path
import render_recorded_landscape as renderer

OUT = Path(__file__).resolve().parents[1] / 'outputs/neon-screen-recording-v4'
REVIEW_FRAMES = {1388,1392,1400,1404,1425,1545,1550,1572,1595,1614,1619,
                1620,4455,4473,4480,4520,4530,4815,4825,4830,4835,4840,
                5525,5530,5560,5610,6072}

if __name__ == '__main__':
    renderer.PREVIEWS |= REVIEW_FRAMES
    sys.argv.extend(['--output-dir', str(OUT), '--rows', str(OUT/'reviewed_rows.jsonl'), '--country-ranking'])
    renderer.main()
