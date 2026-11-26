"""Instrumentation runs only after the isolated virtual action graph is verified."""
import inspect,os,runpy,sys,time,signal
from pathlib import Path
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.graph import get_action_server_names_and_types_by_node
from shared import *
variant=os.environ['VARIANT']
fp=(RUN/'control.jsonl').open('w',buffering=1)
def log(k,**v):write(fp,k,**v)
ctor=Node.__init__
def remap(self,*a,**k):
    k['cli_args']=['--ros-args']+rig.REMAPS
    return ctor(self,*a,**k)
Node.__init__=remap
subscribe_original=Node.create_subscription
def observe_subscription(node, cls, topic, cb, *args, **kwargs):
    if cls.__name__ not in ('ObjectStateArray', 'Detection3DArray'):
        return subscribe_original(node, cls, topic, cb, *args, **kwargs)
    def observed(msg):
        cb(msg)
        info=dict(topic=topic, envelope_ns=ns(msg.header.stamp))
        if cls.__name__=='ObjectStateArray' and msg.states:
            info.update(source_ns=ns(msg.states[0].source_stamp), age_ms=msg.states[0].measurement_age_ms)
        log('state_rx' if cls.__name__=='ObjectStateArray' else 'mask_rx', **info)
    return subscribe_original(node, cls, topic, observed, *args, **kwargs)
Node.create_subscription=observe_subscription
original_client_init=ActionClient.__init__
def private_client(self,node,action_type,action_name,*args,**kwargs):
    assert action_name == '/arm_controller/follow_joint_trajectory'
    assert node.get_namespace() == P
    return original_client_init(self,node,action_type,P+'/trajectory',*args,**kwargs)
ActionClient.__init__=private_client
origwait=ActionClient.wait_for_server
def verify(client,*a,**k):
    ok=origwait(client,*a,**k)
    node=client._node
    servers=[(n,s) for n,space in node.get_node_names_and_namespaces() for s,types in get_action_server_names_and_types_by_node(node,n,space)]
    assert ok and servers==[('virtual_final_bench',P+'/trajectory')],servers
    topics=dict(node.get_topic_names_and_types())
    for topic in ([P+'/detections'] if variant=='A' else [P+'/states',P+'/detections']):
        assert any(i.node_name==node.get_name() for i in node.get_subscriptions_info_by_topic(topic))
    log('isolation',servers=servers,namespace=node.get_namespace());return ok
ActionClient.wait_for_server=verify
send=ActionClient.send_goal_async
def fields(g):
    if variant=='A':return dict(mask_src=getattr(g['self'],'last_src',0))
    out=dict(ident=g.get('identity'))
    for key,(msg,receipt) in g['data'].items():
        if key not in ('joint','mask','state'):continue
        out[key+'_receipt_ns']=int(receipt*1e9);out[key+'_stamp']=ns(msg.header.stamp)
        if key=='state' and msg.states:
            o=msg.states[0];out.update(state_src=ns(o.source_stamp),predicted=o.position_predicted,state_gen=o.generation_id,measurement_age_ms=o.measurement_age_ms)
    return out
def transmit(client,goal,*a,**k):
    node=client._node
    servers=[(name,space,topic) for name,space in node.get_node_names_and_namespaces()
             for topic,types in get_action_server_names_and_types_by_node(node,name,space)]
    assert servers==[('virtual_final_bench',P,P+'/trajectory')],servers
    frame=sys._getframe(1)
    ctx=frame.f_locals if variant=='A' else frame.f_globals
    log('send',trajectory=trajectory_digest(goal),**fields(ctx));return send(client,goal,*a,**k)
ActionClient.send_goal_async=transmit
if variant=='A':
    import real_visual_tracker_final as f
    original=f.FastTracker.detection_cb
    def detection(self,m):
        old=self.target_seen;original(self,m)
        if self.target_seen!=old:self.last_src=ns(m.header.stamp)
    f.FastTracker.detection_cb=detection
    status=f.FastTracker.status
    def report(self,msg):log('A_status',reason=msg);status(self,msg)
    f.FastTracker.status=report
    tick=f.FastTracker.control_cb
    def sample(self):
        if (RUN/'stop').exists():raise KeyboardInterrupt
        before=self.command_count;tick(self)
        log('sample',track=self.target_seen is not None and time.monotonic()-self.target_seen<=f.TARGET_STALE_SEC,mask_src=getattr(self,'last_src',0))
    f.FastTracker.control_cb=sample
    f.main()
else:
    file=Path(os.environ['PROD_NODES']) / 'real_visual_tracker_fused_v2.py'
    text=file.read_text()
    # Insert only observational wrappers after function definitions. They delegate to the unmodified B body.
    key='try:\n\n    print("==================================")'
    assert text.count(key)==1
    text=text.replace(key,"attach_observers(globals())\n\n"+key)
    def attach(g):
        last=0; previous=None; issued=set()
        from v2_input_guard import ReadyEnter
        assert Path(sys.modules['v2_input_guard'].__file__).resolve().parent==file.parent
        assert Path(sys.modules['real_visual_tracker_final'].__file__).resolve().parent==file.parent
        ticket=g['enter_ticket']
        def trace_ticket(name,fn):
            def trace(*args):
                before=ticket.pending
                value=fn(*args)
                if before is not None or ticket.pending is not None or name=='capture':
                    log('ticket_'+name, before=before, after=ticket.pending, result=value)
                return value
            return trace
        for method in ('consume','clear','capture'):
            setattr(ticket, method, trace_ticket(method, getattr(ticket,method)))
        readfd,writefd=os.pipe()
        sys.stdin=os.fdopen(readfd,'r')
        policy=os.environ.get('OPERATOR','aligned')
        original_spin=g['spin']; stopping=False
        def checked_spin():
            nonlocal stopping
            original_spin()
            if not stopping and (RUN/'stop').exists():
                stopping=True
                raise KeyboardInterrupt
        g['spin']=checked_spin
        def watch():
            nonlocal last,previous
            # The virtual operator enters a real newline once per explicit ready prompt.
            ready=g['ready_since'];now_s=time.monotonic()
            case=os.environ['OPERATOR']
            if case in ('early','dry'):
                eligible=ready is None
            else:
                eligible=(ready is not None and not g['armed'] and now_s-ready>.30
                          and .045<g['next_control']-now_s<.075)
            if eligible and not issued:
                issued.add(1)
                if case=='expire':
                    # Delay only the next planner deadline; observation and feedback keep running.
                    g['next_control']+=.13
                os.write(writefd,b'\n')
                log('operator_key',policy=case,ready=ready,until_plan_ms=1000*(g['next_control']-now_s))
                (RUN/'key').touch()
            val=g['enter_original']()
            now=time.monotonic_ns()
            if val:log('enter',ready=g['ready_since'],armed=g['armed'])
            if now-last>=35_000_000 or previous!=g['armed']:
                log('sample',track=g['armed'],**fields(g));last=now;previous=g['armed']
            return val
        g['enter_original']=g['enter_pressed'];g['enter_pressed']=watch
        for name in ['pause_tracking','cancel_current','verify_stop']:
            fn=g[name]
            def wrapped(*a,_name=name,_fn=fn,**k):
                log(_name,reason=str(a[0]) if a else '',armed=g['armed'],**fields(g))
                return _fn(*a,**k)
            g[name]=wrapped
        orig_req=g['require']
        def require(ok,why):
            if not ok and sys._getframe(1).f_code.co_name in ('feedback','controller_ref','check_other_goals'):
                log('feedback_gate',reason=str(why))
            return orig_req(ok,why)
        g['require']=require
    execute_args=[] if os.environ['OPERATOR']=='dry' else ['--execute']
    sys.argv=[str(file),*execute_args,'--ros-args','--log-level','real_visual_tracker_fused_v2:=debug']
    if os.environ['OPERATOR']=='offcycle':
        sys.argv+=['--params-file',str(Path(os.environ['PROD_NODES']) / 'config/v2_filtered_topics.yaml')]

    exec(compile(text,str(file),'exec'),dict(__name__='__main__',__file__=str(file),attach_observers=attach))
