from pathlib import Path
import subprocess,os,time,json,statistics,urllib.request,urllib.error,signal
root=Path(__file__).resolve().parent
out=root/'full-model';out.mkdir(exist_ok=True)
model='~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf'
exe=root/'build/bin/llama-bench'
version=subprocess.run([str(exe),'--version'],text=True,capture_output=True)
(out/'version.txt').write_text(version.stdout+version.stderr)
records=[]
for r in range(3):
 for side in ([0,1] if r%2==0 else [1,0]):
  label='candidate' if side else 'baseline'
  env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(root/('build/bin' if side else 'baseline-bin'))
  cmd=[str(exe),'-m',model,'-fa','on','-p','0','-n','128','-d','2048','-r','1','-o','json','--offline']
  rec={'round':r,'side':label,'command':cmd,'library_path':env['DYLD_LIBRARY_PATH'],'load_before':os.getloadavg(),'start':time.time()}
  print('START',json.dumps(rec),flush=True)
  p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=150)
  rec.update(load_after=os.getloadavg(),seconds=time.time()-rec['start'],returncode=p.returncode)
  (out/f'bench-{r}-{label}.out').write_text(p.stdout)
  (out/f'bench-{r}-{label}.err').write_text(p.stderr)
  if p.returncode:raise RuntimeError((rec,p.stderr[-4000:]))
  data=json.loads(p.stdout);rec['result']=data;records.append(rec)
  (out/'benchmark.json').write_text(json.dumps(records,indent=2))
  print('DONE',json.dumps(rec),flush=True)
ratios=[]
for r in range(3):
 a=next(v for v in records if v['round']==r and v['side']=='baseline')['result'][0]['avg_ts']
 b=next(v for v in records if v['round']==r and v['side']=='candidate')['result'][0]['avg_ts']
 ratios.append(b/a)
print('PAIRED',json.dumps({'speedups':ratios,'median':statistics.median(ratios)}),flush=True)
responses={}
for side in [0,1]:
 label='candidate' if side else 'baseline';port=19871+side
 env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(root/('build/bin' if side else 'baseline-bin'));env['GGML_METAL_FUSION_DEBUG']='1'
 cmd=[str(root/'build/bin/llama-server'),'-m',model,'-fa','on','-c','1024','-np','1','--port',str(port),'--host','127.0.0.1','--offline','--verbose']
 logfile=out/f'server-{label}.log'
 print('SERVER_START',label,os.getloadavg(),flush=True)
 with logfile.open('w') as log:
  p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  try:
   start=time.monotonic()
   while True:
    if p.poll() is not None:raise RuntimeError(f'server {label} exited {p.returncode}: {logfile.read_text()[-3000:]}')
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
      if r.status==200:break
    except (urllib.error.URLError,TimeoutError):pass
    if time.monotonic()-start>90:raise TimeoutError('server load timed out')
    time.sleep(.25)
   payload={'prompt':'Explain in two sentences why the sky looks blue during the day.','n_predict':32,'temperature':0,'seed':1234,'cache_prompt':False,'return_tokens':True,'stream':False}
   req=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=60) as response:data=json.load(response)
   responses[label]=data;(out/f'completion-{label}.json').write_text(json.dumps(data,indent=2))
   print('SERVER_OUTPUT',label,json.dumps({'content':data.get('content'),'tokens':data.get('tokens'),'timings':data.get('timings')}),flush=True)
  finally:
   p.terminate()
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
 print('FUSION_PIPELINE',label,'kernel_cont_mul_mat_f32_k16' in logfile.read_text(),flush=True)
a,b=responses['baseline'],responses['candidate']
check={'content_equal':a['content']==b['content'],'tokens_equal':a['tokens']==b['tokens'],'token_count':len(a['tokens'])}
(out/'equality.json').write_text(json.dumps(check,indent=2));print('EQUALITY',json.dumps(check),flush=True)
if not all(check[k] for k in ['content_equal','tokens_equal']):raise RuntimeError('completion mismatch')
