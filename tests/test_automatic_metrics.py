"""Known geometry and time inputs, not a measurement-accuracy benchmark."""
from copy import deepcopy
import unittest
import numpy as np
from src.sports.volleyball.automatic_metrics import measure_event, attach_metrics, validate_inputs
from tests.test_volleyball_measurements import case, person
from src.core.pose import PoseFrame


class AutomaticMetricsTest(unittest.TestCase):
    def setUp(self):
        self.samples,_,self.metadata=case()
        self.metadata.update(width=640,height=480,nominal_fps=100)
        self.event=dict(kind='stationary_jump',start_frame=0,end_frame=100,
                        takeoff_frame=33,peak_frame=55,landing_frame=77,segment=0)
        self.context=dict(camera='fixed_perpendicular',view='side',time_scale=1.)

    def values(self,event):
        return {v['key']:v for v in event['metrics']}

    def test_no_capture_facts_still_produce_image_measurements(self):
        result=self.values(measure_event(self.samples,self.event,{},self.metadata))
        self.assertGreater(result['image_pelvis_rise_pct']['value'],50)
        self.assertEqual(result['image_lateral_drift_pct']['value'],0)
        self.assertEqual(result['image_trunk_lean_deg']['value'],0)
        self.assertIsNone(result['jump_height_cm']['value'])
        self.assertIsNone(result['pelvis_rise_cm']['value'])

    def test_refined_flight_and_slow_motion_scale(self):
        first=measure_event(self.samples,self.event,self.context,self.metadata)
        h=self.values(first)['jump_height_cm']['value']
        self.assertIsNotNone(h)
        self.assertAlmostEqual(h,9.80665*.5**2/8*100,delta=1.3)
        slow=self.values(measure_event(self.samples,self.event,{**self.context,'time_scale':.5},self.metadata))
        self.assertAlmostEqual(slow['jump_height_cm']['value'],h/4)
        self.assertEqual(first['estimated_contacts'],{'takeoff_frame':30,'landing_frame':80})

    def test_approach_pelvis_cm_is_separate_from_flight_height(self):
        calibration=dict(x1=100,y1=0,x2=100,y2=200,distance_m=2,plane_confirmed=True)
        event={**self.event,'kind':'approach_jump'}
        context={**self.context,'calibration':calibration,'time_scale':None}
        result=self.values(measure_event(self.samples,event,context,self.metadata))
        self.assertIsNone(result['jump_height_cm']['value'])
        hips=[s.points['left_hip'][1] for s in self.samples[32:78]]
        self.assertAlmostEqual(result['pelvis_rise_cm']['value'],hips[0]-min(hips))
        self.assertNotEqual(result['pelvis_rise_cm']['method'],result['jump_height_cm']['method'])
        calibration['plane_confirmed']=False
        rejected=self.values(measure_event(self.samples,event,context,self.metadata))
        self.assertIsNone(rejected['pelvis_rise_cm']['value'])
        self.assertIsNotNone(rejected['image_pelvis_rise_pct']['value'])

    def test_segment_context_and_training_snapshot_are_isolated(self):
        analysis=dict(algorithm='fixture-discovery',segments=[dict(start_frame=0),dict(start_frame=101)],events=[deepcopy(self.event)],warnings=[''])
        inputs=dict(athlete_code='A-17',setup_label='Salon A',recorded_on='2026-09-12',session_phase='after',
                    segments={'101':self.context})
        out=attach_metrics(analysis,self.samples,inputs,self.metadata)
        result=self.values(out['events'][0])
        self.assertIsNone(result['jump_height_cm']['value'])
        context=out['events'][0]['comparison_context']
        self.assertEqual(context['athlete_code'],'A-17')
        self.assertEqual(context['compatibility'],'unverified')
        self.assertEqual(context['capture'],{})

    def test_wrong_context_and_missing_points(self):
        with self.assertRaises(ValueError):
            validate_inputs({'segments':{'999':{}}},self.metadata,[{'start_frame':0}])
        with self.assertRaises(ValueError):
            validate_inputs({'segments':{'0':{'time_scale':float('nan')}}},self.metadata,[{'start_frame':0}])
        samples=list(self.samples); samples[50]=PoseFrame(50,.5,{},'missing')
        result=self.values(measure_event(samples,self.event,self.context,self.metadata))
        self.assertTrue(all(v['value'] is None for v in result.values()))

    def test_calibrated_locomotion_speed_and_missing_time(self):
        samples=[PoseFrame(i,i/100,person(x=100+i)) for i in range(101)]
        event=dict(kind='locomotion',start_frame=0,end_frame=100)
        context={**self.context,'calibration':dict(x1=100,y1=100,x2=200,y2=100,distance_m=10,plane_confirmed=True)}
        result=self.values(measure_event(samples,event,context,self.metadata))
        self.assertAlmostEqual(result['peak_speed_m_s']['value'],10)
        context['time_scale']=None
        self.assertIsNone(self.values(measure_event(samples,event,context,self.metadata))['peak_speed_m_s']['value'])
