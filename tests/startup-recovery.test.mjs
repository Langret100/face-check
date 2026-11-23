import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {SPEECH_PROFILES,chooseKoreanVoice,classifyExpression,cameraSegmentToModel,armRaised,headPoseFromMatrix} from '../interaction-core.js';

const raw=fs.readFileSync(new URL('../main.js',import.meta.url),'utf8');
const flush=async(n=16)=>{for(let i=0;i<n;i++)await Promise.resolve()};

function app(fakeWorker, onGetLandmarker=async()=>{throw Error('Face model missing')}){
 const els=new Map(),timers=new Map();let id=0, mediaStops=0;
 const dom=name=>{
  if(!els.has(name))els.set(name,{id:name,textContent:'',hidden:name==='aiError'||name==='retry',disabled:false,
   classList:{add(){},remove(){},toggle(){}},addEventListener(){},setAttribute(){},
   pause(){},async play(){},querySelectorAll(){return[]}});
  return els.get(name);
 };
 const videoTrack={addEventListener(){},stop(){mediaStops++}};
 const stream={getTracks:()=>[videoTrack],getVideoTracks:()=>[videoTrack]};
 class Q{constructor(){this.x=0;this.y=0;this.z=0;this.w=1}clone(){return new Q()}}
 class V{constructor(){this.x=0;this.y=0;this.z=0}}
 class E{constructor(){this.x=0;this.y=0;this.z=0}}
 const synth={getVoices(){return[]},cancel(){},resume(){},speak(){}};
 const window={devicePixelRatio:1,speechSynthesis:synth,Worker:fakeWorker,createImageBitmap:()=>{},addEventListener(){}};
 const ctx={THREE:{Vector3:V,Quaternion:Q,Euler:E},window,
  document:{getElementById:dom,querySelectorAll:()=>[],hidden:false,addEventListener(){}},
  speechSynthesis:synth,SpeechSynthesisUtterance:class{},performance:{now:()=>0},console,
  Worker:fakeWorker,URL,navigator:{mediaDevices:{getUserMedia:async()=>stream}},
  setTimeout(fn,ms){const seq=++id;timers.set(seq,{fn,ms});return seq},
  clearTimeout(seq){timers.delete(seq)},requestAnimationFrame(){},cancelAnimationFrame(){}};
 Object.assign(ctx,{SPEECH_PROFILES,chooseKoreanVoice,classifyExpression,cameraSegmentToModel,armRaised,headPoseFromMatrix});
 let code=raw.replace(/^import .*?;\s*$/gm,'').replaceAll('import.meta.url',"'https://example.test/demo/'");
 code=code.replace(/model\(\);requestAnimationFrame\(draw\);\s*$/,'');
 code+='\nthis.__h={startCamera,stop,shutdownWorker,initInferenceWorker,withStartupDeadline,getLive:()=>live,getWorkerReady:()=>workerReady};';
 vm.runInNewContext(code,ctx,{timeout:3000});
 ctx.__getLandmarker=onGetLandmarker;
 vm.runInNewContext('modelReady=true;getLandmarker=__getLandmarker;toggleEcho=async()=>{};',ctx);
 return {ctx,dom,timers,h:ctx.__h,get mediaStops(){return mediaStops}};
}

test('v24 regression: Worker init error settles Promise, shows meaningful failure and retry',async()=>{
 let terminations=0;
 class WorkerFail{
  constructor(){this.onmessage=null}
  postMessage(){queueMicrotask(()=>this.onmessage({data:{type:'error',message:'404 models/face_landmarker.task'}}))}
  terminate(){terminations++}
 }
 const a=app(WorkerFail);
 await a.h.startCamera();
 assert.equal(terminations,1,'failed worker must be terminated without a ReferenceError');
 assert.equal(a.h.getLive(),false);
 assert.equal(a.dom('aiError').hidden,false,'error must be visible in the stage');
 assert.match(a.dom('aiErrorMessage').textContent,/Face model missing/);
 assert.match(a.dom('status').textContent,/시작 실패/);
 assert.equal(a.dom('retry').hidden,false,'retry enabled');
 assert.equal(a.mediaStops,1,'camera track closed after error');
});

test('Worker startup watchdog invokes fallback rather than hanging forever',async()=>{
 class SilentWorker{postMessage(){}terminate(){}}
 const a=app(SilentWorker);
 const pending=a.h.startCamera();await flush();
 assert.ok([...a.timers.values()].some(t=>t.ms===18000),'Worker watchdog exists');
 const watchdog=[...a.timers.values()].find(t=>t.ms===18000);
 watchdog.fn();await pending;
 assert.equal(a.dom('aiError').hidden,false);
 assert.equal(a.h.getLive(),false);
 assert.equal(a.timers.size,0,'all Worker and app startup timers cleaned up');
});

test('even a never-settling fallback is bounded, resets error state and allows next retry',async()=>{
 let workerCount=0;
 class Worker{
  constructor(){this.index=++workerCount}
  postMessage(){if(this.index>1)queueMicrotask(()=>this.onmessage({data:{type:'ready'}}))}
  terminate(){}
 }
 const a=app(Worker,()=>new Promise(()=>{}));
 const pending=a.h.startCamera();await flush();
 const deadline=[...a.timers.values()].find(t=>t.ms===42000);
 assert.ok(deadline,'startup deadline prevents an infinite loading screen');
 deadline.fn();await pending;
 assert.equal(a.dom('aiError').hidden,false);
 assert.match(a.dom('aiErrorMessage').textContent,/초과/);
 assert.equal(a.h.getLive(),false);
 await a.h.startCamera();
 assert.equal(a.dom('aiError').hidden,true,'new attempt clears old overlay');
 assert.equal(a.h.getLive(),true,'retry can start tracking after prior hang');
 a.h.stop();
});

test('failed startup probes HEAD only and names missing self-hosted deployment assets',async()=>{
 class WorkerFail{postMessage(){queueMicrotask(()=>this.onmessage({data:{type:'error',message:'WASM missing'}}))}terminate(){}}
 const a=app(WorkerFail);
 const queried=[];
 a.ctx.fetch=async(path,opts)=>{
  queried.push([path,opts.method]);
  return {status:404,ok:false,headers:{get:()=> 'text/html'}};
 };
 await a.h.startCamera();
 assert.equal(queried.length,3);
 assert.ok(queried.every(([,method])=>method==='HEAD'));
 assert.match(a.dom('aiErrorMessage').textContent,/배포 파일 누락/);
 assert.match(a.dom('aiErrorMessage').textContent,/vision_bundle.mjs/);
 assert.match(a.dom('aiErrorMessage').textContent,/face_landmarker.task/);
});

test('FHD and 720p camera requests stay unchanged by the loader repair',()=>{
 const html=fs.readFileSync(new URL('../index.html',import.meta.url),'utf8');
 assert.match(html,/v=25-loader-repair/,'new cache key guarantees latest app script');
 assert.match(html,/id="aiError"/,'a stage-local retry overlay is present');
 assert.match(raw,/width:\{ideal:1280\},height:\{ideal:720\}/);
 assert.match(raw,/1920/);
 assert.doesNotMatch(raw,/workerBusy/,'undeclared workerBusy cannot regress');
});
