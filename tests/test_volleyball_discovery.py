"""Candidate detection tests are not real-video measurement validation."""
import math
import unittest
from src.core.pose import PoseFrame
from src.sports.volleyball.discovery import detect_movements, discover
from tests.test_volleyball_measurements import person


class DiscoveryTest(unittest.TestCase):
    def samples(self, approach=False):
        result = []
        for i in range(150):
            lift = 60*math.sin(math.pi*(i-49)/42) if 50 <= i <= 90 else 0
            points = person(lift=lift)
            dx = min(i,50)*2 if approach else 0
            points = {k:(x+dx,y,s) for k,(x,y,s) in points.items()}
            result.append(PoseFrame(i,i/60,points))
        return result

    def test_jump_without_protocol_or_physical_time(self):
        events = detect_movements(self.samples(),60)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['kind'],'stationary_jump')
        self.assertLess(events[0]['takeoff_frame'],events[0]['peak_frame'])
        self.assertLess(events[0]['peak_frame'],events[0]['landing_frame'])
        self.assertEqual(events[0]['metrics'],[])
        approach = detect_movements(self.samples(True),60)
        self.assertEqual(approach[0]['kind'],'approach_jump')

    def test_still_and_crouch_are_not_jump(self):
        samples = [PoseFrame(i,i/60,person()) for i in range(150)]
        self.assertEqual(detect_movements(samples,60),[])
        for i in range(50,90):
            p = dict(samples[i].points)
            for k,(x,y,s) in p.items():
                if 'hip' in k or 'shoulder' in k: p[k]=(x,y+25,s)
            samples[i]=PoseFrame(i,i/60,p)
        self.assertEqual(detect_movements(samples,60),[])

    def test_cut_and_missing_frame_do_not_bridge_flight(self):
        samples=self.samples()
        result=discover(samples,{'nominal_fps':60},[70],[])
        self.assertEqual(result['events'],[])
        self.assertEqual(len(result['segments']),2)
        samples[70]=PoseFrame(70,70/60,{},'missing')
        self.assertEqual(detect_movements(samples,60),[])

    def test_good_segment_survives_bad_segment(self):
        samples=self.samples()
        bad=[PoseFrame(i,i/60,{},'missing') for i in range(150,180)]
        result=discover(samples+bad,{'nominal_fps':60},[150],[])
        self.assertEqual(len(result['events']),1)
        self.assertEqual(result['segments'][1]['missing_frames'],30)

    def test_lateral_rigid_shift_is_not_called_running(self):
        samples=[]
        for i in range(100):
            p={k:(x+i*3,y,s) for k,(x,y,s) in person().items()}
            samples.append(PoseFrame(i,i/60,p))
        self.assertEqual(detect_movements(samples,60),[])

    def test_adaptive_cut_detection_and_regular_motion(self):
        import numpy as np
        from src.adapters.video_review import ShotBoundaryDetector
        detector=ShotBoundaryDetector()
        for n in range(20):
            self.assertFalse(detector.update(np.full((192,108,3),n,dtype=np.uint8)))
        self.assertTrue(detector.update(np.full((192,108,3),120,dtype=np.uint8)))
        self.assertFalse(detector.update(np.full((192,108,3),121,dtype=np.uint8)))

    def test_articulated_horizontal_motion_produces_locomotion_candidate(self):
        samples=[]
        for i in range(150):
            p={k:(x+i*2,y,s) for k,(x,y,s) in person().items()}
            for side,sign in [('left',1),('right',-1)]:
                for part in ('ankle','big_toe','small_toe','heel'):
                    x,y,s=p[side+'_'+part]
                    p[side+'_'+part]=(x+sign*20*math.sin(i*.3),y,s)
            samples.append(PoseFrame(i,i/60,p))
        found=detect_movements(samples,60)
        self.assertTrue(found)
        self.assertTrue(all(e['kind']=='locomotion' for e in found))
