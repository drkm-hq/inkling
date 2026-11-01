from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for name,cmd in [
 ('graph-order-tests.log',[str(root/'build/bin/test-metal-graph-optimize')]),
 ('test-relative-hardened.log',[str(root/'build/bin/test-backend-ops'),'test','-b','MTL0','-o','CONT_MUL_MAT']),
 ('relative-order-probe.log',[sys.executable,str(root/'probe_relative_order.py')]),
]:
 print('RUN',name,flush=True)
 with (root/name).open('w') as f: result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print('RESULT',name,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
