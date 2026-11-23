"""Chromium smoke test of the real v25 startup and recovery code + UI.
Mocks 3D initialisation, MediaPipe, and camera so an AI init failure can be
forced and recovery verified. Not a real GPU or face-tracking benchmark.
"""
import pathlib, re
from playwright.sync_api import sync_playwright
root=pathlib.Path(__file__).resolve().parents[1]
source=(root/'main.js').read_text()
source=re.sub(r'^import .*?;\s*$', '',source,flags=re.M)
source=source.replace('model();requestAnimationFrame(draw);', "modelReady=true;getLandmarker=async()=>{throw Error('Face model fallback unavailable')};toggleEcho=async()=>{};startCamera();")
source=source.replace('import.meta.url',"'https://example.test/demo/'")
source='''
class V3{constructor(){this.x=0;this.y=0;this.z=0}}
class Q{constructor(){this.x=0;this.y=0;this.z=0;this.w=1}}
class E{constructor(){this.x=0;this.y=0;this.z=0}}
const THREE={Vector3:V3,Quaternion:Q,Euler:E};
const SPEECH_PROFILES={squirrel:{},anime:{}};
const chooseKoreanVoice=()=>null,classifyExpression=()=>null,cameraSegmentToModel=()=>null,armRaised=()=>false,headPoseFromMatrix=()=>null;
''' + source

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--use-gl=angle'])
    page=browser.new_page(viewport={'width':1366,'height':768})
    page.add_init_script('''
      window.__workerInstances=0;
      window.Worker=class SimWorker{
        constructor(){this.attempt=++window.__workerInstances}
        postMessage(msg){ if(msg.type==='init') queueMicrotask(()=>this.onmessage({data:this.attempt===1?
            {type:'error',message:'Face model asset 404'}:{type:'ready'}})); }
        terminate(){}
      };
      const getUserMedia=async()=>new MediaStream();
      Object.defineProperty(navigator,'mediaDevices',{value:{getUserMedia},configurable:true});
      HTMLMediaElement.prototype.play=async function(){};
    ''')
    # No external HTTP access in the sandbox. Exercise the DOM + startup code in-place.
    html=(root/'index.html').read_text()
    html=re.sub(r'<script type=\"module\" src=\"\./main\.js\?[^\"]+\"></script>', '',html)
    page.set_content(html)
    page.evaluate('''() => {
      window.__workerInstances=0;
      window.Worker=class SimWorker{constructor(){this.attempt=++window.__workerInstances}postMessage(msg){if(msg.type==='init')queueMicrotask(()=>this.onmessage({data:this.attempt===1?{type:'error',message:'Face model asset 404'}:{type:'ready'}}))}terminate(){}};
      Object.defineProperty(navigator,'mediaDevices',{value:{getUserMedia:async()=>new MediaStream()},configurable:true});
      HTMLMediaElement.prototype.play=async function(){};
    }''')
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.add_script_tag(content=source)
    page.wait_for_function('!document.querySelector("#aiError").hidden',timeout=5000)
    first=page.locator('#aiErrorMessage').inner_text()
    assert 'Face model fallback unavailable' in first,first
    assert '얼굴 인식' in page.locator('#aiError strong').inner_text()
    assert page.locator('#retry').is_visible()
    page.locator('#aiRetry').click()
    page.wait_for_function('document.querySelector("#status").textContent.includes("실시간 표정 인식 중")',timeout=5000)
    assert page.locator('#aiError').is_hidden()
    assert page.locator('#buildVersion').inner_text().startswith('v25')
    assert not errors,errors
    print('CHROMIUM BROWSER TEST PASS: first AI init 404 -> visible error -> retry -> running state')
    print('first error:',first[:220])
    print('status:',page.locator('#status').inner_text())
    browser.close()

