"""Render the readable duet layout and reviewed turn handovers."""
import sys
from pathlib import Path
import render_recorded_landscape as renderer
from render_recording_review_v4 import REVIEW_FRAMES as V4
from render_recording_review_v5 import REVIEW_FRAMES as V5
OUT=Path(__file__).resolve().parents[1]/'outputs/neon-screen-recording-v6'
REVIEW_FRAMES={280,295,310,325,340,350,680,684,687,690,1768,2353,2358,2363,
 2458,2466,2468,2470,2474,2476,2477,2480,2485,2490,
 2677,2681,2686,2691,2725,2729,2733,2737,2741,2764,2770,2772,2788,2792,2880,
 4115,4125,4130,4135,4140,4145,4146,4150,4155,4160,4165,4170,4175,4179,
 4180,4200,4220,4240,4260,4280,4825,4830,4837,4840}
if __name__=='__main__':
    renderer.PREVIEWS|=V4|V5|REVIEW_FRAMES
    sys.argv.extend(['--output-dir',str(OUT),'--rows',str(OUT/'reviewed_rows.jsonl'),
      '--country-ranking','--music-start-seconds','2'])
    renderer.main()
