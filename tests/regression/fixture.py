import os,sys,time,threading,subprocess,signal,csv,json,heapq
from pathlib import Path
import numpy as np
import rclpy
from rclpy.action import CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile,DurabilityPolicy
from sensor_msgs.msg import JointState
from control_msgs.msg import DynamicJointState,InterfaceValue,JointTrajectoryControllerState
from action_msgs.msg import GoalStatusArray
from object_state_msgs.msg import ObjectState,ObjectStateArray
from vision_msgs.msg import Detection3DArray,Detection3D,ObjectHypothesisWithPose
from shared import *
variant=os.environ['VARIANT'];dataset=os.environ['DATASET']
fp=(RUN/'rig.jsonl').open('w',buffering=1)
def emit(k,**v):write(fp,k,**v)
def pt(out,a):out.x,out.y,out.z=map(float,a)
class Bench(rig.VirtualRig):
    def __init__(self):
        self.start=None;self.inject_joint=False
        original=rig.ActionServer
        def cancel(handle):
            emit('cancel_request',goal_id=bytes(handle.goal_id.uuid).hex())
            return CancelResponse.ACCEPT
        def server(*a,**k):return original(*a,cancel_callback=cancel,**k)
        rig.ActionServer=server
        try:super().__init__()
        finally:rig.ActionServer=original
        self.hw=self.create_publisher(DynamicJointState,P+'/hardware',10)
        self.ctrl=self.create_publisher(JointTrajectoryControllerState,P+'/controller',10)
        self.states=self.create_publisher(ObjectStateArray,P+'/states',10)
        self.initial=self.create_publisher(GoalStatusArray,P+'/trajectory/_action/status',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.initial.publish(GoalStatusArray())
        self.prev=(time.monotonic(),self.q.copy())
    def accept(self,handle):
        super().accept(handle)
        emit('goal',count=self.goal_count,trajectory=trajectory_digest(handle.request),goal_id=bytes(handle.goal_id.uuid).hex(),cmd=list(handle.request.trajectory.points[0].positions),q=self.q.tolist())
    def execute(self,h):
        res=rig.FollowJointTrajectory.Result()
        while not self.stop_event.is_set():
            with self.lock:
                now=time.monotonic()
                if self.segment is None or self.segment[0] is not h:
                    h.abort();res.error_code=-1;return res
                self.advance(now)
                if h.is_cancel_requested:
                    self.segment=None;h.canceled();emit('canceled',goal_id=bytes(h.goal_id.uuid).hex());return res
                if now>=self.segment[3]+self.segment[4]:
                    self.segment=None;h.succeed();emit('succeeded');return res
            time.sleep(.004)
        h.abort();res.error_code=-1;return res
    def update(self):
        with self.lock:
            now=time.monotonic();self.advance(now)
            prev,q0=self.prev;v=(self.q-q0)/max(.0001,now-prev)
            self.prev=now,self.q.copy()
            stamp=self.get_clock().now().nanoseconds
            j=JointState();j.name=rig.JOINTS;j.position=self.q.tolist();j.velocity=v.tolist()
            setstamp(j.header.stamp,stamp-(250_000_000 if self.inject_joint else 0));self.joint_pub.publish(j)
            h=DynamicJointState();setstamp(h.header.stamp,stamp)
            for i in range(5):
                h.joint_names.append('dxl'+str(i+1));it=InterfaceValue()
                it.interface_names=['Torque Enable','Present Position','Present Velocity','Present Current','Present Input Voltage']
                it.values=[1.,float(self.q[i]) if i<4 else 0.,float(v[i]) if i<4 else 0.,0.,12.]
                h.interface_values.append(it)
            self.hw.publish(h)
            c=JointTrajectoryControllerState();setstamp(c.header.stamp,stamp);c.joint_names=rig.JOINTS
            # Output is the eased trajectory's current reference. Feedback is the ideal virtual actuator.
            c.output.positions=self.q.tolist();c.feedback.positions=self.q.tolist();self.ctrl.publish(c)
            if self.start is not None:
                emit('joint',elapsed=now-self.start,joint_src=ns(j.header.stamp),q=list(j.position),v=list(j.velocity),tcp=rig.ARM.fk(self.q)[:3,3].tolist())
def mask(src,items):
    m=Detection3DArray();setstamp(m.header.stamp,src);m.header.frame_id=FRAME
    for tid,cls,score,x,y,z in items:
        d=Detection3D();d.header=m.header;d.id=tid;h=ObjectHypothesisWithPose();h.hypothesis.class_id=cls;h.hypothesis.score=float(score)
        pt(h.pose.pose.position,[x,y,z]);d.results.append(h);m.detections.append(d)
    return m
def publish(r,item,base,gen='fixture-life-1'):
    arrival,_,kind,src,t,phase=item
    absent=phase in ('short_hide','lost')
    original_t=(src-base)/1e9
    pw=truth(original_t);pc=rig.R.T@(pw-rig.T)
    tid='target_1'
    if kind=='mask':
        if phase=='other_identity':tid='other'
        m=mask(src,[] if absent else [[tid,'blue block',.91,*pc]])
        r.det_pub.publish(m);emit('mask',src=src,phase=phase,items=len(m.detections),ident=tid,orig=src)
    else:
        s=ObjectStateArray();s.header.frame_id=FRAME;pub=r.get_clock().now().nanoseconds;setstamp(s.header.stamp,pub)
        if not absent:
            o=ObjectState();o.class_id='blue block';o.track_id=tid;o.generation_id=gen
            o.state=1;o.confidence=.92;o.valid_3d=True;o.stale=False;o.missed=0
            o.measurement_source_type=ObjectState.SOURCE_FAST_BBOX_DEPTH;o.measurement_source='fast_bbox_depth'
            setstamp(o.source_stamp,src);setstamp(o.last_seen,src)
            o.measurement_age_ms=max(0.,(pub-src)/1e6)
            pt(o.position,pc);pt(o.position_std,[.003]*3);pt(o.velocity,rig.R.T@((truth(original_t+.001)-truth(original_t-.001))/.002))
            if phase=='bad_prediction':
                o.position_predicted=True;pt(o.velocity,[float('nan'),0.,0.])
            s.states.append(o)
            if getattr(r,'ambiguous',False):
                from copy import deepcopy
                other=deepcopy(o);other.generation_id='fixture-life-2'
                s.states.append(other)
        r.states.publish(s);emit('state',generation=gen,src=src,pub=pub,phase=phase,prediction_target_ns=None,items=len(s.states))
def main():
    rclpy.init(args=[]);r=Bench();ex=MultiThreadedExecutor(num_threads=6);ex.add_node(r)
    thread=threading.Thread(target=ex.spin,daemon=True);thread.start()
    out=(RUN/'stdout.log').open('w')
    proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).with_name('control.py'))],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            output=(RUN/'stdout.log').read_text()
            if ('ARM ACTION SERVER = PASS' if variant=='A' else 'START TCP') in output:break
            if proc.poll() is not None:raise RuntimeError(output[-2000:])
            time.sleep(.05)
        else:raise RuntimeError('startup timeout '+output[-1500:])
        r.start=time.monotonic()+.1;base=r.get_clock().now().nanoseconds+100_000_000
        emit('start',mono=int(r.start*1e9),ros=base,dataset=dataset)
        from stimulus import run
        run(r,base,os.environ['OPERATOR'],publish,emit,r.start,RUN)
        assert thread.is_alive(), 'virtual executor stopped unexpectedly'
        emit('end',goals=r.goal_count,rejected=r.rejected_count);r.inject_joint=False
        if os.environ['OPERATOR']!='joint':
            assert proc.poll() is None, 'controller exited before end of stimulus'
    finally:
        if proc.poll() is None:
            (RUN/"stop").touch()
            try:proc.wait(timeout=7)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        out.close();r.stop_event.set();ex.shutdown(timeout_sec=3);thread.join(2)
        r.destroy_node()

        rclpy.shutdown()
    assert proc.returncode==0
    if variant!='A': assert 'TRACKING SESSION CLOSED' in (RUN/'stdout.log').read_text()
if __name__=='__main__':main()
