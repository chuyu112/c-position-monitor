"""Render V5 with the user-confirmed two-second music pre-roll."""
import sys
from pathlib import Path
import render_recorded_landscape as renderer
from render_recording_review_v4 import REVIEW_FRAMES as V4_REVIEW_FRAMES
OUT=Path(__file__).resolve().parents[1]/'outputs/neon-screen-recording-v5'
REVIEW_FRAMES={0,30,59,60,61,89,3347,3367,3373,3377,3383,3387,3391,3397,3400,
 3518,3522,3523,3524,3525,3540,3827,3837,3847,3852,3867,3887,3911,3926,3929,
 3930,3934,3941,3954,3957,3958,3967,3982,3994,4018,4042,4066,4070,4071}
if __name__=='__main__':
    renderer.PREVIEWS|=V4_REVIEW_FRAMES|REVIEW_FRAMES
    sys.argv.extend(['--output-dir',str(OUT),'--rows',str(OUT/'reviewed_rows.jsonl'),
                     '--country-ranking','--music-start-seconds','2'])
    renderer.main()
