import json
from math import dist

def check(folder, case):
    def events(file):
        return [json.loads(line) for line in (folder/file).read_text().splitlines()]
    c=events('control.jsonl')
    r=events('rig.jsonl')
    def matches(rows, name):
        return [x for x in rows if x['event']==name]
    sends=matches(c,'send')
    goals=matches(r,'goal')
    joints=matches(r,'joint')
    keys=matches(c,'operator_key')
    pauses=[x for x in matches(c,'pause_tracking') if x['armed']]
    moved=max(dist(joints[0]['tcp'],x['tcp']) for x in joints)
    stdout=(folder/'stdout.log').read_text()
    assert matches(c,'isolation') and len(sends)==len(goals)
    assert matches(r,'end')[-1]['rejected']==0
    assert [x['trajectory'] for x in sends]==[x['trajectory'] for x in goals], 'wire trajectory mismatch'
    detail={}
    if case!='final':
        assert len(keys)==1
    if case in ('dry','early','identity'):
        assert not goals
        if case=='identity':
            assert any(x['after'] for x in matches(c,'ticket_capture'))
            assert not any(x['result'] for x in matches(c,'ticket_consume'))
        if case=='dry':
            assert 'EXECUTE: False' in stdout and 'DRY TRACK' in stdout
        if case=='early':
            assert keys[0]['ready'] is None and 'TARGET READY:' in stdout
    else:
        assert len(goals)>=1 and moved>.001, 'no effective motion'
    if case in ('offcycle','async','repeated','consume','joint'):
        assert .0<keys[0]['until_plan_ms']<100
        assert sum(bool(x['result']) for x in matches(c,'ticket_consume'))==1
    if case in ('async','repeated'):
        assert stdout.count('CONTINUOUS TRACKING ENABLED')==1
        assert len(matches(c,'mask_rx'))>=75
        assert len(matches(c,'state_rx'))>=150
        assert sends[-1]['t']-sends[0]['t']>4e9 and not pauses
        sources=[x['source_ns'] for x in matches(c,'state_rx') if 'source_ns' in x]
        if case=='repeated':
            states=[x for x in matches(c,'state_rx') if 'source_ns' in x]
            masks=matches(c,'mask_rx')
            assert len(states)==160 and len(masks)==80
            assert len(set(sources))==80 and sum(a==b for a,b in zip(sources,sources[1:]))==80
            for a,b in zip(states[::2],states[1::2]):
                assert a['source_ns']==b['source_ns'] and a['age_ms']<b['age_ms']<=100
            for x in states:
                actual=(x['envelope_ns']-x['source_ns'])/1e6
                assert 0<=actual and abs(actual-x['age_ms'])<.001
            duration=(states[-1]['t']-states[0]['t'])/1e9
            detail.update(unique_sources=80,repeats=80,state_hz=159/duration,measurement_hz=79/((states[-2]['t']-states[0]['t'])/1e9))
    if case=='consume':
        hide=next(x['t'] for x in r if x.get('phase')=='short_hide')
        recovered=next(x['t'] for x in r if x['t']>hide and x.get('phase')=='sync')
        assert stdout.count('TARGET READY:')>=2
        assert pauses and recovered>pauses[0]['t']>=hide and sends[-1]['t']<pauses[0]['t']
    if case=='joint':
        gates=[x for x in matches(c,'feedback_gate') if 'JOINT TIMESTAMP' in x['reason']]
        assert 'FAULT LATCHED: JOINT TIMESTAMP' in stdout and 'NO AUTOMATIC RESTART' in stdout
        assert gates and sends[-1]['t']<gates[0]['t']
        assert any('joint' in x['reason'].lower() for x in gates)
    if case in ('joint','track_identity'):
        fault,=matches(r,'inject')
        canceled=[x for x in matches(r,'canceled') if x['goal_id']==fault['goal_id']]
        requests=[x for x in matches(r,'cancel_request') if x['goal_id']==fault['goal_id']]
        assert len(requests)==len(canceled)==1
        assert sends[-1]['t']<fault['t']<requests[0]['t']<=canceled[0]['t']
        assert matches(r,'end')[0]['t']-canceled[0]['t']>1e9
        assert stdout.count('CONTINUOUS TRACKING ENABLED')==1
        if case=='joint':
            assert 'REQUESTING CANCEL' in stdout and 'STOP NOT VERIFIED: JOINT TIMESTAMP' in stdout
        else:
            assert pauses and pauses[0]['reason']=='NO TARGET OR LOCK LOST'
            assert "TARGET READY: ('target_1', 'fixture-life-2')" in stdout
            assert any(x.get('track') and x['t']<fault['t'] for x in matches(c,'sample'))
        detail.update(canceled_goal=fault['goal_id'])
    return dict(**detail,goals=len(goals),feedback_samples=len(joints),max_tcp_displacement_mm=round(moved*1000,3),pauses=len(pauses))
