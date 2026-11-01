from pathlib import Path
import subprocess,os,time,json,urllib.request,urllib.error
root=Path(__file__).resolve().parent
out=root/'relative-order-probe';out.mkdir(exist_ok=True)
model='~/.cache/huggingface/hub/models--unsloth--Inkling-Small-GGUF/snapshots/1a19ef82883cb7b9c581b93c30ea252dabbf658d/UD-IQ3_XXS/Inkling-Small-UD-IQ3_XXS-00001-of-00003.gguf'
env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(root/'build/bin');env['GGML_METAL_FUSION_DEBUG']='1'
cmd=[str(root/'build/bin/llama-server'),'-m',model,'-fa','on','-c','1024','-np','1','--port','19873','--host','127.0.0.1','--offline','--verbose']
logfile=out/'server.log';print('START',json.dumps({'cmd':cmd,'load':os.getloadavg()}),flush=True)
with logfile.open('w') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  start=time.monotonic()
  while True:
   if p.poll() is not None:raise RuntimeError(f'server exited {p.returncode}: {logfile.read_text()[-3000:]}')
   try:
    with urllib.request.urlopen('http://127.0.0.1:19873/health',timeout=2) as r:
     if r.status==200:break
   except (urllib.error.URLError,TimeoutError):pass
   if time.monotonic()-start>90:raise TimeoutError('server load timed out')
   time.sleep(.25)
  payload={'prompt':'Explain in two sentences why the sky looks blue during the day.','n_predict':32,'temperature':0,'seed':1234,'cache_prompt':False,'return_tokens':True,'stream':False}
  (out/'request.json').write_text(json.dumps(payload,indent=2))
  req=urllib.request.Request('http://127.0.0.1:19873/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=60) as response: data=json.load(response)
  (out/'completion.json').write_text(json.dumps(data,indent=2))
 finally:
  p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
baseline=json.loads((root/'full-model/completion-baseline.json').read_text())
summary={'pipeline_seen':'kernel_cont_mul_mat_f32_k16' in logfile.read_text(),'content_equal':data['content']==baseline['content'],'tokens_equal':data['tokens']==baseline['tokens'],'token_count':len(data['tokens']),'load_after':os.getloadavg()}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('RESULT',json.dumps(summary),flush=True)
assert summary['pipeline_seen'] and summary['content_equal'] and summary['tokens_equal'],summary
