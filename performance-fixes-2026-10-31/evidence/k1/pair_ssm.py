from pathlib import Path
import subprocess,os,statistics,json,sys
root=Path(__file__).resolve().parent
args=sys.argv[1:] or ['168','0','4','1','1','100']
values={0:[],1:[]};checks=[]
for r in range(7):
 for side in ([0,1] if r%2==0 else [1,0]):
  env=os.environ.copy(); env['DYLD_LIBRARY_PATH']=str(root/('build/bin' if side else 'baseline-bin'))
  p=subprocess.run([str(root/'bench_ssm'),*args],env=env,text=True,capture_output=True)
  (root/f'ssm-{r}-{side}.log').write_text(p.stdout+p.stderr)
  if p.returncode:raise SystemExit(f'failed {r} {side} {p.returncode}: {p.stderr[-3000:]}')
  check=next(l for l in p.stdout.splitlines() if l.startswith('CHECK'));checks.append(check)
  times=[float(l.split('us=')[1]) for l in p.stdout.splitlines() if l.startswith('TIME')]
  value=statistics.median(times);values[side].append(value)
  print(f'ROUND {r} candidate={side} median_us={value} {check}',flush=True)
assert len(set(checks))==1,checks
ratios=[a/b for a,b in zip(values[0],values[1])]
print(json.dumps({'args':args,'baseline_us':values[0],'candidate_us':values[1],'paired_speedup':ratios,'median_speedup':statistics.median(ratios),'median_saving_us':statistics.median([a-b for a,b in zip(values[0],values[1])])},indent=2))
