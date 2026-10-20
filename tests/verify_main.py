"""Real main.py: hk_env, port 8000, HTTP and browser regression verification."""
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import hashlib,json,secrets,socket,subprocess,time,uuid,re
import cv2
import httpx
import numpy as np
from core.selective_crypto import GROUPS,inspect_chunk,read_video
from core.selective_recognition import synthetic_frame
import config
from utils.aes_crypto import encrypt_file

run=ROOT/'tests/runtime'/('main-'+uuid.uuid4().hex);run.mkdir(parents=True,mode=0o700)
with socket.socket() as check:
 check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);check.bind(('127.0.0.1',8000))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',SRX_TEST_MODE='synthetic',SRX_TEST_FRAMES='24',SRX_DATA_ROOT=str(run),SRX_PORT='8000',MPLCONFIGDIR=str(run/'mpl'),XDG_CACHE_HOME=str(run/'cache'))
legacy=run/'legacy-fixtures/2026-09-15/00/chunk_0001';legacy.mkdir(parents=True)
raw=legacy/'synthetic.bin';raw.write_bytes(b''.join(synthetic_frame(i).tobytes() for i in range(4)))
password=secrets.token_hex(20);encrypt_file(str(raw),str(legacy/'chunk_0001.enc'),password)
(legacy/'chunk_0001.json').write_text(json.dumps({'width':320,'height':240,'channels':3,'fps':10,'complete':True,'frame_count':4}))
report={'command':[sys.executable,str(ROOT/'main.py')],'address':'http://127.0.0.1:8000','run':str(run),'synthetic':True}
log=(run/'main.log').open('w');proc=subprocess.Popen(report['command'],cwd=ROOT,env=env,stdout=log,stderr=log)
try:
 with httpx.Client(base_url=report['address'],timeout=20,trust_env=False) as c:
  for _ in range(150):
   if proc.poll() is not None:raise RuntimeError((run/'main.log').read_text())
   try:r=c.get('/');r.raise_for_status();break
   except httpx.TransportError:time.sleep(.1)
  else:raise RuntimeError('startup timeout')
  assert 'href="/selective/"' not in r.text
  assert c.get('/selective/').status_code==401
  assert c.get('/selective/api/chunks').status_code==401
  assert c.get('/recordings/regression.txt').status_code==404  # 로그인 없는 정적 파일 경로는 제거됨
  assert c.get('/snapshot/raw').status_code==401
  assert c.get('/decryption_page').status_code==302
  login=c.post('/api/admin/login',json={'id':config.ADMIN_ID,'password':config.ADMIN_PASSWORD});login.raise_for_status()
  for path in ('/register','/employees','/assets','/decryption_page'):
   page=c.get(path);page.raise_for_status();assert '<html' in page.text.lower()
  page=c.get('/selective/');page.raise_for_status();csrf=re.search("const csrf='([^']+)'",page.text)[1];headers={'X-SRX-CSRF':csrf}
  for _ in range(150):
   status=c.get('/selective/api/status').json()
   if not status['recording']:break
   time.sleep(.1)
  assert status['error'] is None,status
  assert c.get('/snapshot/raw').status_code==200
  assert c.get('/snapshot/cam_0').status_code==200
  chunks=c.get('/selective/api/chunks').json();assert len(chunks)==3
  assert sum(x['format']=='selective' for x in c.get('/api/assets').json())==3
  # Choose the last captured chunk, where all stable classifications have warmed up.
  cid=max(chunks,key=lambda item:inspect_chunk(run/'chunks'/item['id'])[0]['manifest']['meta']['frames'][0]['capture_id'])['id']
  doc,masks=inspect_chunk(run/'chunks'/cid);meta=doc['manifest']['meta']
  protected=read_video(run/'chunks'/cid/'protected.avi',len(masks),meta['shape'])
  def wait(jid):
   for _ in range(250):
    j=c.get('/selective/api/jobs/'+jid).json()
    if j['status']!='running':return j
    time.sleep(.03)
   raise RuntimeError('restore timeout')
  checks=[]
  for group in (None,*GROUPS,'wrong'):
   key=(run/'keys'/(group+'.key')).read_bytes() if group in GROUPS else (secrets.token_bytes(32) if group else b'')
   response=c.post('/selective/api/restore/'+cid,headers=headers,content=key);response.raise_for_status();jid=response.json()['job'];j=wait(jid)
   if group=='wrong':assert j['status']=='failed' and not (run/'results'/jid).exists()
   else:
    assert j['status']=='complete' and j['completed_at'] and j['chunk']==cid
    avi=c.get('/selective/api/jobs/'+jid+'/download');avi.raise_for_status();saved=run/(str(group)+'.avi');saved.write_bytes(avi.content)
    frames=read_video(saved,len(masks),meta['shape'])
    for i,frame in enumerate(frames):
     allowed=np.zeros(frame.shape[:2],bool)
     for g in (GROUPS if group=='master' else ([group] if group else [])):allowed|=masks[i][g]
     original=synthetic_frame(meta['frames'][i]['capture_id'])
     assert np.array_equal(frame[allowed],original[allowed])
     assert np.array_equal(frame[~allowed],original[~allowed] if group=='master' else protected[i][~allowed])
    assert c.get('/selective/api/jobs/'+jid+'/frame/0').content.startswith(b'\x89PNG')
    assert c.get('/selective/api/jobs/'+jid+'/frame/-1').status_code==400
   checks.append({'key':group or 'none','status':j['status'],'job':jid})
  assert c.post('/selective/api/restore/'+cid,content=b'').status_code==403
  mismatch=c.post('/selective/api/restore/'+cid,headers=headers|{'X-SRX-Key-Role':'master'},content=(run/'keys/external.key').read_bytes())
  assert mismatch.status_code==403 and '일치하지 않습니다' in mismatch.json()['detail']
  assert c.post('/selective/api/restore/'+cid,headers=headers,content=b'x'*33).status_code==413
  assert c.post('/selective/api/restore/invalid',headers=headers).status_code==400
  with httpx.Client(base_url=report['address'],trust_env=False) as other:
   assert other.get('/selective/').status_code==401
   other.post('/api/admin/login',json={'id':config.ADMIN_ID,'password':config.ADMIN_PASSWORD}).raise_for_status()
   other.get('/selective/')
   assert other.get('/selective/api/jobs/'+checks[1]['job']+'/download').status_code==404
  # Actual web failure on a tampered ciphertext, then restore only this generated test fixture.
  path=run/'chunks'/cid/'external.gcm';before=path.read_bytes()
  path.write_bytes(bytes([before[0]^1])+before[1:])
  bad=c.post('/selective/api/restore/'+cid,headers=headers,content=(run/'keys/master.key').read_bytes()).json()['job']
  assert wait(bad)['status']=='failed' and not (run/'results'/bad).exists();path.write_bytes(before)
  # Both the current full-original AES-GCM path and historical AES path remain usable.
  assert c.get('/api/admin/chunks').status_code==200
  current=c.post('/api/admin/verify_chunk',data={'folder_path':str(run/'chunks'/cid),'password':config.ADMIN_PASSWORD});current.raise_for_status();current_job=current.json()['job_id']
  for _ in range(400):
   current_state=c.get('/api/admin/restore/status/'+current_job).json()
   if current_state['status'] in ('completed','failed'):break
   time.sleep(.05)
  assert current_state['status']=='completed',current_state
  assert c.get('/api/admin/restore/result/'+current_job).status_code==200
  response=c.post('/api/admin/verify_chunk',data={'folder_path':str(legacy),'password':password});response.raise_for_status();legacy_job=response.json()['job_id']
  for _ in range(200):
   j=c.get('/api/admin/restore/status/'+legacy_job).json()
   if j['status'] in ('completed','failed'):break
   time.sleep(.05)
  assert j['status']=='completed',j
  result=c.get('/api/admin/restore/result/'+legacy_job);result.raise_for_status();lp=run/'historical-restored.mp4';lp.write_bytes(result.content)
  assert len(read_video(lp,4,[240,320,3]))==4
  report.update(policy=checks,recording=status['stats'],legacy_restore=True,current_original_restore=True,http_security=True)
 # Exercise the actual HTML form and image player in Chromium.
 from playwright.sync_api import sync_playwright
 os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(ROOT/'tests/runtime/browsers')
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
  page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.request.post(report['address']+'/api/admin/login',data={'id':config.ADMIN_ID,'password':config.ADMIN_PASSWORD})
  page.goto(report['address']+'/employees');page.get_by_role('link',name='선택적 복원',exact=True).click();page.wait_for_selector('#chunks .chunk',state='attached')
  page.locator('[data-id="'+cid+'"]').click()
  page.wait_for_function("document.getElementById('preview').naturalWidth === 320")
  page.locator('input[name=role][value=external]').check();page.set_input_files('#key',str(run/'keys/external.key'))
  page.get_by_role('button',name='선택적 복원 실행').click()
  page.wait_for_function("!document.getElementById('resultBox').classList.contains('hidden')")
  page.wait_for_function("document.getElementById('result').naturalWidth === 320")
  with page.expect_download() as download:page.get_by_role('link',name='무손실 AVI 다운로드').click()
  download.value.save_as(str(run/'browser-restored.avi'))
  assert len(read_video(run/'browser-restored.avi',len(masks),meta['shape']))==len(masks)
  page.get_by_role('button',name='재생 / 일시정지').click();page.wait_for_timeout(350)
  assert int(page.locator('#frame').input_value())>0
  page.screenshot(path=str(run/'browser.png'),full_page=True)
  assert not errors,errors
  browser.close();report['browser_form_playback_download']=True
 report['success']=True
finally:
 proc.terminate()
 try:proc.wait(timeout=20)
 except subprocess.TimeoutExpired:proc.kill();proc.wait()
 log.close();report['server_stopped']=True
 (run/'report.json').write_text(json.dumps(report,indent=2))
 (ROOT/'tests/runtime/latest-main.txt').write_text(str(run))
 print(json.dumps(report,ensure_ascii=False))
