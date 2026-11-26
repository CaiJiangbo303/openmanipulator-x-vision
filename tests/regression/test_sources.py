import ast
import hashlib
import json
import os
from pathlib import Path
from v2_input_guard import ReadyEnter

ws=Path(os.environ["SOURCE_ROOT"])
for name,expected in json.loads((ws.parent/"sources.json").read_text())["files"].items():
    data=(ws/name).read_bytes()
    assert hashlib.sha256(data).hexdigest()==expected,name
    if name.endswith(".py"): ast.parse(data,name)
gate=ReadyEnter(.1)
ident=('t','g')
for dt,ok in [(0,True),(.1,True),(.100001,False)]:
    gate.capture(ident,-1,0)
    assert gate.consume(ident,-1,dt)==ok
    assert not gate.consume(ident,-1,dt)
print('Snapshot hashes, syntax, 100 ms boundary and single consumption passed')
