import json,os,time
from pathlib import Path
import rig_fixture as rig
import numpy as np
# Verify the modules actually imported by the production controller.
from hashlib import sha256
kin_dir=Path(os.environ['KIN_PARENT'])/'manipulator_grasp'
expected=json.loads(Path(__file__).with_name('kinematics.json').read_text())
loaded={}
for filename, digest in expected.items():
    module='manipulator_grasp' + ('' if filename=='__init__.py' else '.'+filename[:-3])
    from importlib import import_module
    path=Path(import_module(module).__file__).resolve()
    assert path==(kin_dir/filename).resolve() and sha256(path.read_bytes()).hexdigest()==digest,path
    loaded[module]=str(path)
(Path(os.environ['RUN'])/('imports_'+str(os.getpid())+'.json')).write_text(json.dumps(loaded,indent=2))
P=os.environ['R4_PREFIX']
assert P.startswith('/r4_virtual_') and os.environ['ROS_DOMAIN_ID']==os.environ['TEST_DOMAIN'] and os.environ['ROS_LOCALHOST_ONLY']=='1'
RUN=Path(os.environ['RUN'])
FRAME='camera_color_optical_frame'
rig.REMAPS=[s.replace(rig.PREFIX,P) for s in rig.REMAPS]
rig.PREFIX=P;rig.FRAME=FRAME
rig.REMAPS += ['-r', '__ns:='+P]
for real,end in [('/bench/timestamp_guard/states','states'),('/bench/real_self_filter/detections_3d','detections'),('/object_state_manager/states','states'),('/object_self_filter/detections_3d_mask','detections'),('/arm_controller/controller_state','controller'),('/dynamic_joint_states','hardware')]:
    rig.REMAPS+=['-r',real+':='+P+'/'+end]
def ns(t): return t.sec*10**9+t.nanosec
def setstamp(st,x): st.sec,st.nanosec=divmod(int(x),10**9)
def write(fp,event,**kw):
    fp.write(json.dumps(dict(t=time.monotonic_ns(),event=event,**kw),separators=(',',':'))+'\n')
def truth(t):
    # Gentle 8-second motion plus fixed offset; all positions reachable from Q0.
    if os.environ.get("OPERATOR") == "final":
        return rig.TCP0 + np.array([0.,.020,0.])
    return rig.TCP0+np.array([.009*np.sin(t*2*np.pi/8),.013*np.cos(t*2*np.pi/8),0.])


def trajectory_digest(goal):
    from hashlib import sha256
    from rclpy.serialization import serialize_message
    return sha256(serialize_message(goal.trajectory)).hexdigest()
