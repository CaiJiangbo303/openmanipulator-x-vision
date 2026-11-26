"""Export an explicit source selection, build it, then run short ROS regressions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def invoke(command, env, logfile, timeout):
    with logfile.open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            result = child.wait(timeout=timeout)
        finally:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            for _ in range(20):
                groups = subprocess.check_output(['ps','-eo','pgid='],text=True).split()
                if str(child.pid) not in groups:
                    break
                time.sleep(.05)
            else:
                raise RuntimeError(f'process group {child.pid} survived cleanup')
    if result:
        content = logfile.read_text(errors='replace')
        if any(token in content for token in ('ModuleNotFoundError', 'PackageNotFoundError', 'command not found','Could not find a package configuration file','No module named')):
            raise FileNotFoundError('dependency missing: '+str(logfile))
        raise RuntimeError(f'exit {result}: {logfile}')

def prepare(args, ws):
    repo=args.repo.resolve()
    if not (Path('/opt/ros/humble/setup.bash').is_file() and shutil.which('colcon')):
        raise FileNotFoundError('ROS Humble and colcon required')
    rev=args.revision if args.mode=='BASELINE' else 'HEAD'
    commit=subprocess.check_output(['git','-C',repo,'rev-parse','--verify',rev+'^{commit}'],text=True).strip()
    if args.mode=='BASELINE' and commit!=args.revision:
        raise ValueError('use a full immutable commit ID')
    records={}
    for name in args.inputs.read_text().splitlines():
        rel=Path(name)
        if not name or rel.is_absolute() or '..' in rel.parts or name in records:
            raise ValueError('invalid or duplicate input '+name)
        if args.mode=='BASELINE':
            proc=subprocess.run(['git','-C',repo,'show',commit+':'+name],capture_output=True)
            if proc.returncode:
                raise FileNotFoundError(name+' absent from '+commit)
            data=proc.stdout
        else:
            source=(repo/rel).resolve()
            if not source.is_relative_to(repo):
                raise ValueError('input escapes repository')
            data=source.read_bytes()
        target=ws/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        records[name]=digest(data)
    save(ws.parent/'sources.json',dict(mode=args.mode,commit=commit,root=str(repo),files=records))
    if args.level!='L0':
        if not args.kinematics:
            raise FileNotFoundError('--kinematics requires the approved manipulator_grasp directory')
        expected=json.loads((HERE/'kinematics.json').read_text())
        actual={k:digest((args.kinematics/k).read_bytes()) for k in expected}
        save(ws.parent/'kinematics.json',dict(directory=str(args.kinematics.resolve()),files=actual))
        if actual!=expected or args.kinematics.name!='manipulator_grasp':
            raise FileNotFoundError('approved kinematics package/hash unavailable')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['BASELINE','CURRENT'])
    p.add_argument('level',choices=['L0','L1','ALL'])
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--revision')
    p.add_argument('--inputs',type=Path,default=HERE/'inputs.txt')
    p.add_argument('--contract',choices=['legacy','generation'],required=True)
    p.add_argument('--kinematics',type=Path)
    motion_cases=('final','offcycle','early','async','repeated','identity','track_identity','consume','dry','joint')
    p.add_argument('--case', choices=motion_cases, help='run one isolated L1 motion case')
    p.add_argument('--site-setup',type=Path)
    args=p.parse_args()
    if (args.mode=='BASELINE') != bool(args.revision) or (args.revision and not re.fullmatch('[0-9a-f]{40}',args.revision)):
        p.error('BASELINE requires --revision FULL_SHA; CURRENT reads the selected working files')
    if args.case and args.level!='L1':
        p.error('--case requires L1')
    out=args.output.resolve()
    if out.is_relative_to(args.repo.resolve()) or (args.kinematics and out.is_relative_to(args.kinematics.resolve())):
        p.error('output must be outside source directories')
    out.mkdir(parents=True,exist_ok=False)
    ws=out/'export'
    rows=[]
    shell=dict(PATH='/usr/bin:/bin',HOME=os.environ['HOME'],LANG='C.UTF-8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    domains=iter(random.sample(range(20,190),32))
    def record(name,fn):
        start=time.monotonic()
        try:
            detail=fn()
            status='PASS'
        except FileNotFoundError as exc:
            status,detail='BLOCKED',str(exc)
        except Exception as exc:
            status,detail='FAIL',repr(exc)
        rows.append(dict(name=name,status=status,detail=detail,seconds=round(time.monotonic()-start,3)))
        save(out/'results.json',rows)
        print(name,status,detail or '',flush=True)
        (out/'summary.txt').write_text('\n'.join(f"{r['name']} {r['status']} {r['detail'] or ''}" for r in rows)+'\n')
        return status=='PASS'
    def case(name,script,motion=False,site=False):
        folder=out/name
        folder.mkdir()
        env=dict(shell,MATRIX='C' if site else 'isolated',SITE_SETUP=str(args.site_setup or ''),
                 SOURCE_ROOT=str(args.repo.resolve() if site else ws),RUN=str(folder),
                 KIN_PARENT=str(args.kinematics.resolve().parent) if args.kinematics else str(ws),
                 HARNESS=str(HERE),TEST_DOMAIN=str(next(domains)),CONTRACT=args.contract,
                 R4_PREFIX='/r4_virtual_'+uuid.uuid4().hex[:12],OPERATOR=name,
                 VARIANT='A' if name=='final' else 'B',DATASET='synthetic')
        def execute(program,log):
            invoke(['bash',HERE/'environment.sh','python3','-B',program],env,folder/log,45)
        def body():
            if site and not args.site_setup.is_file():
                raise FileNotFoundError(args.site_setup)
            execute(HERE/'graph.py','before.log')
            try:
                execute(script,'output.log')
                if motion:
                    from invariants import check
                    return check(folder,name)
            finally:
                execute(HERE/'graph.py','after.log')
        return record(name,body)
    if not record('export',lambda:prepare(args,ws)):
        return 2
    if not record('build',lambda:invoke(['bash',HERE/'build.sh',ws],shell,out/'build.log',640)):
        return 2
    if not case('interfaces',HERE/'check_interfaces.py'):
        return 1
    if args.site_setup:
        case('site_interfaces',HERE/'check_interfaces.py',site=True)
    if args.level in ('L0','ALL'):
        case('sources',HERE/'test_sources.py')
        case('state_gate',ws/'tests/test_vision_v2_state_gate.py')
    if args.level in ('L1','ALL'):
        if not args.case:
            case('generation',ws/'tests/test_object_state_generation.py')
            case('chain',HERE/'test_chain.py')
        for name in ([args.case] if args.case else motion_cases):
            case(name,HERE/'fixture.py',True)
    return int(any(r['status']!='PASS' for r in rows))

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    sys.exit(main())
