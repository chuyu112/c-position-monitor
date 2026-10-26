"""Render the AIJING correction while retaining c7 presentation settings."""
import sys
from pathlib import Path
import render_recorded_landscape as renderer
from render_recording_review_v7 import REVIEW_FRAMES as V7,V6,V4,V5
OUT=Path(__file__).resolve().parents[1]/'outputs/neon-screen-recording-v8'
REVIEW_FRAMES={4058,4065,4070,4071,4075,4080,4084,4085,4088,4090,4092,4095,
              4100,4104,4105,4108,4110,4114,4115,4120}
if __name__=='__main__':
 renderer.PREVIEWS|=V4|V5|V6|V7|REVIEW_FRAMES
 sys.argv.extend(['--output-dir',str(OUT),'--rows',str(OUT/'reviewed_rows.jsonl'),
   '--country-ranking','--hide-ranking-footer','--music-start-seconds','2'])
 renderer.main()
