"""Render the corrected 1:17 duet without the leaderboard footer."""
import sys
from pathlib import Path
import render_recorded_landscape as renderer
from render_recording_review_v6 import REVIEW_FRAMES as V6,V4,V5
OUT=Path(__file__).resolve().parents[1]/'outputs/neon-screen-recording-v7'
REVIEW_FRAMES={2288,2290,2293,2294,2296,2300,2304,2307,2310,2311,2315,2320,2330,2340,2358}
if __name__=='__main__':
 renderer.PREVIEWS|=V4|V5|V6|REVIEW_FRAMES
 sys.argv.extend(['--output-dir',str(OUT),'--rows',str(OUT/'reviewed_rows.jsonl'),
   '--country-ranking','--hide-ranking-footer','--music-start-seconds','2'])
 renderer.main()
