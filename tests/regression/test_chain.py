"""Exercise the four installed nodes over DDS, then replay a mismatched pair."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
import rclpy
from rclpy.action.graph import get_action_names_and_types
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from rosidl_runtime_py.convert import message_to_ordereddict
from object_state_msgs.msg import ObjectState,ObjectStateArray,ObjectPredictionArray
from four_dof_feasibility_msgs.msg import FourDofFeasibilityArray
from target_selector_msgs.msg import TargetSelection
from std_msgs.msg import String
from object_motion_predictor.predictor_node import ObjectMotionPredictor
from four_dof_feasibility.feasibility_node import FourDofFeasibilityNode,position
from target_selector.selector_node import TargetSelectorNode
from grasp_command_bridge.grasp_command_bridge_node import GraspCommandBridge

def ns(stamp):return stamp.sec*10**9+stamp.nanosec
def stamp(t,n):t.sec,t.nanosec=divmod(n,10**9)
def point(p):return np.array([p.x,p.y,p.z])

def main():
    candidate=os.environ['CONTRACT']=='generation'
    prefix=os.environ['R4_PREFIX']
    topics=['/object_state_manager/states','/object_motion_predictor/predictions','/four_dof_feasibility/results','/target_selector/selection','/grasp_command_bridge/status']
    types=[ObjectStateArray,ObjectPredictionArray,FourDofFeasibilityArray,TargetSelection,String]
    args=['--ros-args','-r','__ns:='+prefix]
    for topic in topics+['/tf','/tf_static']:args+=['-r',topic+':='+prefix+topic]
    rclpy.init(args=args)
    observer=Node('chain_test')
    predictor=ObjectMotionPredictor();feasible=FourDofFeasibilityNode()
    selector=TargetSelectorNode();bridge=GraspCommandBridge()
    nodes=[observer,predictor,feasible,selector,bridge]
    assert bridge.dry_run
    ex=SingleThreadedExecutor()
    for node in nodes:ex.add_node(node)
    received=[[] for _ in topics]
    phase='initial'
    trace=(Path(os.environ['RUN'])/'wire.jsonl').open('w',buffering=1)
    def record(i,msg):
        received[i].append(msg)
        print(json.dumps(dict(phase=phase,mono_ns=time.monotonic_ns(),topic=topics[i],msg=message_to_ordereddict(msg))),file=trace)
    subs=[observer.create_subscription(typ,topic,lambda msg,i=i:record(i,msg),10) for i,(topic,typ) in enumerate(zip(topics,types))]
    pubs=[observer.create_publisher(typ,topic,10) for typ,topic in zip(types[:3],topics[:3])]
    def spin(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:ex.spin_once(timeout_sec=.005)
    seen_source={}
    gen1,gen2=str(uuid.uuid4()),str(uuid.uuid4())
    base_target=position([.10,.25,-.35,.12])
    translation=np.array([.032,-.011,.007])
    world_target=base_target+translation
    def inject(generation):
        now=observer.get_clock().now().nanoseconds
        msg=ObjectStateArray();msg.header.frame_id='world';stamp(msg.header.stamp,now)
        obj=ObjectState();obj.class_id='blue block';obj.track_id='track_001'
        obj.generation_id=generation;obj.state=ObjectState.TRACKED;obj.valid_3d=True;obj.confidence=.95
        obj.position.x,obj.position.y,obj.position.z=map(float,world_target)
        obj.position_std.x=obj.position_std.y=obj.position_std.z=.003
        stamp(obj.source_stamp,now-32_000_000);stamp(obj.last_seen,now-24_000_000)
        obj.measurement_age_ms=24.;obj.measurement_source_type=ObjectState.SOURCE_PRECISE_MASK_DEPTH
        msg.states=[obj];seen_source[now]=(generation,ns(obj.source_stamp))
        pubs[0].publish(msg);spin(.08)
    result={}
    try:
        tf=TransformStamped();tf.header.frame_id='world';tf.child_frame_id='arm_link1'
        tf.header.stamp=observer.get_clock().now().to_msg();tf.transform.rotation.w=1.
        tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z=map(float,translation)
        broadcaster=StaticTransformBroadcaster(observer);broadcaster.sendTransform(tf)
        spin(.5)
        assert get_action_names_and_types(observer)==[]
        for _ in range(10):inject(gen1)
        spin(.15)
        assert all(received),'missing stage '+str([len(x) for x in received])
        assert any(m.data.startswith('ACCEPT ') for m in received[-1]),[m.data for m in received[-1]]
        for m in received[1]:
            pred=m.predictions[0];g,src=seen_source[ns(m.header.stamp)]
            assert ns(pred.source_stamp)==(src if candidate else ns(m.header.stamp))
            if candidate:assert pred.generation_id==g
        for stage in (2,3):
            for msg in received[stage]:
                for item in (msg.candidates if stage==2 else [msg]):
                    assert msg.header.frame_id=='world'
                    assert np.allclose(point(item.target_arm),base_target) and np.allclose(point(item.target_world),world_target)
                    if candidate:
                        g,src=seen_source[ns(msg.header.stamp)]
                        assert item.generation_id==g and ns(item.source_stamp)==src
        old=deepcopy(received[3][-1]);phase='change_generation'
        mark=len(received[4])
        for _ in range(6):inject(gen2)
        statuses=[v.data for v in received[4][mark:]]
        assert any('generation changed' in s for s in statuses)==candidate,statuses
        assert any(s.startswith('ACCEPT ') for s in statuses)
        # Original, unmodified old selection through DDS after the new generation locks.
        sel_pub=observer.create_publisher(TargetSelection,topics[3],10)
        spin(.08);phase='old_generation';mark=len(received[4]);sel_pub.publish(old);spin(.08)
        reply=[v.data for v in received[4][mark:]]
        assert reply and not any(s.startswith('ACCEPT ') for s in reply)
        if candidate:assert any('generation changed' in s for s in reply),reply
        # Feed an independent selector input pair. Remove the prediction subscriber
        # producing new feasibility results so only the deliberately wrong pair competes.
        ex.remove_node(feasible);feasible.destroy_node();nodes.remove(feasible)
        pair=deepcopy(received[1][-1]);bad=deepcopy(received[2][-1])
        paired_feasible=deepcopy(bad)
        pair.header.stamp=observer.get_clock().now().to_msg()
        bad.header=deepcopy(pair.header)
        for c in bad.candidates:c.track_id='wrong-track'
        phase='wrong_track';mark=len(received[3]);pubs[1].publish(pair);pubs[2].publish(bad);spin(.15)
        assert len(received[3])>mark and not received[3][-1].valid
        if candidate:
            phase='wrong_generation';bad=deepcopy(paired_feasible)
            pair.header.stamp=observer.get_clock().now().to_msg();bad.header=deepcopy(pair.header)
            for c in bad.candidates:c.track_id='track_001';c.generation_id=gen1
            mark=len(received[3]);pubs[1].publish(pair);pubs[2].publish(bad);spin(.15)
            assert len(received[3])>mark and not received[3][-1].valid
        assert not get_action_names_and_types(observer)
        result=dict(bridge='ACCEPT',stage_counts=list(map(len,received)),generation_preserved=candidate,source_preserved=candidate,wrong_track_rejected=True,wrong_generation_rejected=candidate,old_selection_status=reply,generation_switch=statuses,arm_xyz=base_target.tolist(),world_xyz=world_target.tolist(),action_endpoints=[])
        print(json.dumps(result));(Path(os.environ['RUN'])/'chain.json').write_text(json.dumps(result,indent=2))
    finally:
        trace.close();ex.shutdown()
        for node in reversed(nodes):node.destroy_node()
        rclpy.shutdown()
if __name__=='__main__':main()
