"""Source-frame overlays and optional constant-PTS browser preview."""
import json
import shutil
import subprocess
import cv2
import numpy as np
from src.core.pose import PoseFrame, point

EDGES = [('left_shoulder','right_shoulder'), ('left_hip','right_hip')]
EDGES += [(side+'_'+a, side+'_'+b) for side in ('left','right')
          for a,b in [('shoulder','hip'),('hip','knee'),('knee','ankle'),('ankle','big_toe'),('ankle','heel')]]


def overlay(frame, people, label=''):
    frame = frame.copy()
    for i, person in enumerate(people):
        sample = PoseFrame(0, None, person)
        valid = [point(sample,k) for k in person if point(sample,k) is not None]
        for a,b in EDGES:
            pa,pb = point(sample,a),point(sample,b)
            if pa is not None and pb is not None:
                cv2.line(frame, tuple(map(int,pa)), tuple(map(int,pb)), (0,220,100), 3)
        for p in valid:
            cv2.circle(frame, tuple(map(int,p)), 4, (0,220,255), -1)
        if valid:
            x,y = np.min(valid,axis=0).astype(int)
            cv2.putText(frame, f'Sporcu {i+1}', (x,max(25,y-12)), cv2.FONT_HERSHEY_SIMPLEX, .8,(0,220,255),2)
    if label:
        cv2.rectangle(frame,(0,0),(frame.shape[1],42),(25,25,25),-1)
        cv2.putText(frame,label,(12,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    return frame


def render_preview(source, pose_path, destination, metadata, events):
    times = metadata.get('timestamps', [])
    if not shutil.which('ffmpeg') or len(times) < 2 or any(t is None for t in times):
        return None
    delta = np.diff(times)
    if np.min(delta) <= 0 or np.max(np.abs(delta-np.mean(delta))) > .0001:
        return None  # Preserve VFR source playback instead of inventing preview timing.
    fps = 1/float(np.mean(delta))
    height = min(720, metadata['height'])
    height -= height % 2
    width = max(2, round(metadata['width']*height/metadata['height'])//2*2)
    capture = cv2.VideoCapture(str(source))
    process = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24',
                                '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an',
                                '-c:v','libx264','-preset','veryfast','-pix_fmt','yuv420p',
                                '-movflags','+faststart',str(destination)],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with pose_path.open() as data:
            for row in data:
                sample = json.loads(row)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Önizleme kaynağı beklenenden kısa.')
                label = 'Hareket taramasi'
                for n,event in enumerate(events):
                    if event['start_frame'] <= sample['frame'] <= event['end_frame']:
                        label = f'Aday {n+1}'
                        if 'takeoff_frame' in event:
                            label += ' - Ucus adayi' if event['takeoff_frame'] <= sample['frame'] < event['landing_frame'] else ' - Yaklasma / inis'
                        break
                points = [sample['points']] if not sample['issue'] else []
                painted = overlay(frame, points, label)
                process.stdin.write(cv2.resize(painted,(width,height)).tobytes())
        process.stdin.close()
        error = process.stderr.read()
        if process.wait() != 0:
            raise OSError(error.decode(errors='replace')[:500])
    finally:
        capture.release()
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:
            process.stdin.close()
        process.stderr.close()
    return destination
