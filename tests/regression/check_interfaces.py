"""Compare ordered .msg schemas to the imported ROS classes and node field uses."""
import ast
import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path
from ament_index_python.packages import get_package_prefix
from rclpy.serialization import serialize_message, deserialize_message
from rosidl_runtime_py.utilities import get_message

MESSAGES = {
    'object_state_msgs': ['ObjectState','ObjectStateArray','ObjectPrediction','ObjectPredictionArray'],
    'four_dof_feasibility_msgs': ['FourDofFeasibility','FourDofFeasibilityArray'],
    'target_selector_msgs': ['TargetSelection'],
}
# Bind names within each production function to the message classes they denote.
NODES = {
 'object_motion_predictor.predictor_node': {'state_callback': {'msg':'ObjectStateArray','state':'ObjectState','prediction':'ObjectPrediction','output':'ObjectPredictionArray'}},
 'four_dof_feasibility.feasibility_node': {'process_prediction': {'msg':'ObjectPredictionArray','prediction':'ObjectPrediction','result':'FourDofFeasibility','output':'FourDofFeasibilityArray'}},
 'target_selector.selector_node': {'find_prediction':{'candidate':'FourDofFeasibility','prediction':'ObjectPrediction','predictions':'ObjectPredictionArray'},'select_target':{'output':'TargetSelection','candidate':'FourDofFeasibility','prediction':'ObjectPrediction','predictions':'ObjectPredictionArray','feasibility':'FourDofFeasibilityArray'}},
 'grasp_command_bridge.grasp_command_bridge_node': {k:{'msg':'TargetSelection'} for k in ('selection_identity','validate_selection','process_selection','publish_decision','selection_callback')},
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def schema(path,pkg):
    fields=[]
    for line in path.read_text().splitlines():
        line=line.partition('#')[0].strip()
        if not line or '=' in line:continue
        typ,name=line.split()
        base=typ.split('[')[0]
        canon={'float32':'float','float64':'double','bool':'boolean'}.get(base,base)
        if '/' not in canon and canon[0].isupper():canon=pkg+'/'+canon
        suffix=typ[len(base):]
        fields.append((name,'sequence<'+canon+'>' if suffix=='[]' else canon+suffix))
    return fields

def sample(cls):
    msg=cls()
    for field,typ in cls.get_fields_and_field_types().items():
        seq=typ.startswith('sequence<')
        item=typ[9:-1] if seq else typ.split('[')[0]
        if '/' in item:
            pkg,name=item.split('/')
            value=sample(get_message(pkg+'/msg/'+name))
        elif item=='string':value='test'
        elif item=='boolean':value=True
        elif item in ('float','double'):value=.25
        else:value=1
        if seq:value=[value]
        elif '[' in typ:value=[value]*int(re.search(r'\[(\d+)\]',typ)[1])
        setattr(msg,field,value)
    return msg

def main():
    ws=Path(os.environ['SOURCE_ROOT'])
    label=os.environ['MATRIX']
    install=ws/('vision_ws/install' if label=='C' else 'install')
    out=Path(os.environ['RUN'])
    env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','LANG','PYTHONPATH','LD_LIBRARY_PATH','CMAKE_PREFIX_PATH','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH','RMW_IMPLEMENTATION','ROS_DOMAIN_ID','ROS_LOCALHOST_ONLY','ROS_DISTRO','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','ROS_LOG_DIR')}
    deps=set(MESSAGES)|{p.split('.')[0] for p in NODES}|set('std_msgs builtin_interfaces geometry_msgs control_msgs trajectory_msgs tf2_ros rclpy rosidl_runtime_py vision_msgs action_msgs sensor_msgs object_state_manager'.split())
    origins={p:get_package_prefix(p) for p in sorted(deps)}
    rows=[]; allfields={}
    def contained(p):return Path(p).resolve().is_relative_to(install.resolve())
    if label!='C':
        for k in ('AMENT_PREFIX_PATH','CMAKE_PREFIX_PATH','COLCON_PREFIX_PATH','LD_LIBRARY_PATH','PYTHONPATH'):
            for path in env.get(k,'').split(':'):
                if path and '/vision_ws/install' in path:raise RuntimeError('unexpected site overlay '+path)
        for p in deps:
            if p in MESSAGES or p in {n.split('.')[0] for n in NODES} or p=='object_state_manager':assert contained(origins[p]),origins[p]
    for pkg,names in MESSAGES.items():
        for name in names:
            src=ws/'vision_ws/src'/pkg/'msg'/(name+'.msg')
            cls=get_message(pkg+'/msg/'+name)
            path=Path(importlib.import_module(cls.__module__).__file__)
            expected=schema(src,pkg); actual=list(cls.get_fields_and_field_types().items())
            msg=sample(cls); blob=serialize_message(msg)
            decoded=deserialize_message(blob,cls)
            assert serialize_message(decoded)==blob and decoded==msg
            if label!='C':assert contained(path),path
            allfields[name]=dict(actual)
            rows.append(dict(name=name,kind='message',source=str(src),source_sha=sha(src),loaded=str(path),loaded_sha=sha(path),expected=expected,actual=actual,wire_bytes=len(blob),issues=[] if actual==expected else ['ordered schema mismatch']))
    for module,bindings in NODES.items():
        path=Path(importlib.import_module(module).__file__)
        pkg,filename=module.split('.');src=ws/'vision_ws/src'/pkg/pkg/(filename+'.py')
        issues=[]; uses=[]
        if sha(path)!=sha(src):issues.append('loaded/source SHA differ')
        if label!='C':assert contained(path),path
        for kind,fp in [('loaded',path),('source',src)]:
            for fn in ast.walk(ast.parse(fp.read_bytes())):
                if not isinstance(fn,ast.FunctionDef) or fn.name not in bindings:continue
                for attr in ast.walk(fn):
                    if not isinstance(attr,ast.Attribute) or not isinstance(attr.value,ast.Name):continue
                    typ=bindings[fn.name].get(attr.value.id)
                    if typ is None:continue
                    uses.append((kind,fn.name,attr.lineno,typ,attr.attr))
                    if attr.attr not in allfields[typ]:issues.append(f'{kind} L{attr.lineno} {typ}.{attr.attr}')
        assert uses
        rows.append(dict(name=module,kind='node',source=str(src),source_sha=sha(src),loaded=str(path),loaded_sha=sha(path),uses=uses,issues=issues))
    issues=[r['name'] for r in rows if r['issues']]
    report=dict(environment=label,compatible=not issues,issues=issues,rows=rows,env=env,package_prefix=origins,sys_path=sys.path)
    (out/'matrix.json').write_text(json.dumps(report,indent=2))
    print('COMPATIBLE' if not issues else 'INCOMPATIBLE',issues)
    return 3 if issues else 0
if __name__=='__main__':sys.exit(main())
