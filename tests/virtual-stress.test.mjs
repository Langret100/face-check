import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {SPEECH_PROFILES,chooseKoreanVoice,classifyExpression,cameraSegmentToModel,armRaised,headPoseFromMatrix} from '../interaction-core.js';

const getSource=()=>fs.readFileSync(new URL('../main.js',import.meta.url),'utf8');
const flush=async(n=8)=>{for(let i=0;i<n;i++)await Promise.resolve()};
function appHarness(){
 const els=new Map(),speech=[],windowEvents={},clock={now:0};
 let idx=0;
 const timers=new Map(), voices=[{name:'Microsoft SunHi',lang:'ko-KR',localService:true}];
 const dom=id=>{if(!els.has(id))els.set(id,{id,textContent:'',hidden:false,disabled:false,checked:false,
  classList:{add(){},remove(){},toggle(){}},addEventListener(){},setAttribute(){},
  pause(){},async play(){},querySelectorAll(){return[]}});return els.get(id)};
 const radios=[{value:'squirrel',disabled:false,checked:true,addEventListener(){}},{value:'anime',disabled:false,checked:false,addEventListener(){}}];
 class Q{constructor(){this.x=0;this.y=0;this.z=0;this.w=1}clone(){return new Q()}}
 class V{constructor(){this.x=0;this.y=0;this.z=0}}
 class E{constructor(){this.x=0;this.y=0;this.z=0}}
 const synth={getVoices(){return voices},cancel(){},resume(){},speak(u){speech.push(u)}};
 const ctx={THREE:{Vector3:V,Quaternion:Q,Euler:E},document:{getElementById:dom,querySelectorAll:s=>s.includes('name=avatar')?radios:[],hidden:false,addEventListener(){}},
  window:{devicePixelRatio:1,speechSynthesis:synth,addEventListener(type,fn){windowEvents[type]=fn}},
  speechSynthesis:synth,SpeechSynthesisUtterance:class{constructor(text){this.text=text}},performance:{now:()=>clock.now},console,
  setTimeout(fn,ms){const id=++idx;timers.set(id,{fn,ms});return id},clearTimeout(id){timers.delete(id)},
  requestAnimationFrame(){},cancelAnimationFrame(){},setInterval(){},clearInterval(){},
  navigator:{mediaDevices:{getUserMedia:async()=>{throw Error('stub camera unset')}}},
  GLTFLoader:class{loadAsync(){return Promise.resolve({id:'anime'})}},URL};
 Object.assign(ctx,{SPEECH_PROFILES,chooseKoreanVoice,classifyExpression,cameraSegmentToModel,armRaised,headPoseFromMatrix});
 let code=getSource().replace(/^import .*?;\s*$/gm,'').replaceAll('import.meta.url',"'http://localhost/'");
 code=code.replace(/model\(\);requestAnimationFrame\(draw\);\s*$/,'');
 code+='\nthis.__h={draw,voiceSay,applyFaceResult,applyPoseResult,updateVoice,switchAvatar,stop,startCamera,stopInference,beginInference,echo,voiceState,setLive:x=>{live=x},setAvatar:x=>{selectedAvatar=x},setModelReady:x=>{modelReady=x},setSquirrel:x=>{squirrelScene=x},getCurrent:()=>selectedAvatar,getBound:()=>boundAvatarKind,setRender:(r,c)=>{renderer=r;camera=c;scene={}},setCharacter:x=>{character=x},setFaceSeen:x=>{hadFace=x},setHeadRig:(head,neck)=>{headBone=head;neckBone=neck},getTarget:()=>({...target}),getCurrentPose:()=>({...current}),setWorker:(w,ready)=>{aiWorker=w;workerReady=ready},getWorkerState:()=>({session:workerFrameId,busy:workerFrameInFlight,token:aiPerf.token})};';
 vm.runInNewContext(code,ctx,{timeout:3000});
 return {h:ctx.__h,ctx,els,radios,speech,windowEvents,clock,timers,dom};
}
const face=(scores={jawOpen:0})=>({blendScores:scores,headPoints:{left:{x:.3,y:.4},right:{x:.7,y:.4},nose:{x:.5,y:.5}}});

test('60-Hz synthetic render simulation: 3,600 frames preserve native FHD buffer without resolution degradation',()=>{
 const {h,dom,ctx,clock}=appHarness();
 const canvas=dom('three');
 let ratio=0,rendered=0,resizeCount=0;
 const renderer={setPixelRatio(v){ratio=v},setSize(w,height){resizeCount++;canvas.width=Math.floor(w*ratio);canvas.height=Math.floor(height*ratio)},render(){rendered++}};
 h.setRender(renderer,{});
 canvas.clientWidth=1920;canvas.clientHeight=1080;ctx.window.devicePixelRatio=1;
 for(let frame=0;frame<3600;frame++){clock.now=frame*1000/60;h.draw(clock.now)}
 assert.equal(rendered,3600);
 assert.equal(resizeCount,1,'no per-frame framebuffer resize');
 assert.equal(canvas.width,1920);assert.equal(canvas.height,1080);
 assert.match(dom('fps').textContent,/60 fps|61 fps/);
 assert.match(dom('fps').title,/렌더 버퍼 1920×1080/);
});

test('high-DPI and portrait settings render at display resolution up to FHD, independent of FPS',()=>{
 for(const [w,height,dpr,expectedW,expectedH] of [[1280,720,2,1920,1080],[320,200,2,640,400],[1080,1920,1,1080,1920],[2560,1440,1,1920,1080]]){
  const {h,ctx,clock,dom}=appHarness();const canvas=dom('three');canvas.clientWidth=w;canvas.clientHeight=height;ctx.window.devicePixelRatio=dpr;
  let scale=1;h.setRender({setPixelRatio(v){scale=v},setSize(x,y){canvas.width=Math.floor(x*scale);canvas.height=Math.floor(y*scale)},render(){}},{});
  h.draw(clock.now);
  assert.ok(Math.abs(canvas.width-expectedW)<=1,`width for ${w}x${height}@${dpr} = ${canvas.width}`);
  assert.ok(Math.abs(canvas.height-expectedH)<=1);
 }
});

test('camera requests native 1280x720 high-definition video without runtime downscale',async()=>{
 const {h,ctx,dom}=appHarness();let requested;
 const videoTrack={addEventListener(){},stop(){}};
 ctx.navigator.mediaDevices.getUserMedia=async constraints=>{requested=constraints;return {getTracks:()=>[videoTrack],getVideoTracks:()=>[videoTrack]}};
 h.setModelReady(true);
 vm.runInNewContext('initInferenceWorker=async()=>{workerReady=true;return true};toggleEcho=async()=>{};',ctx);
 await h.startCamera();
 assert.equal(requested.video.width.ideal,1280);
 assert.equal(requested.video.height.ideal,720);
 assert.equal(requested.video.frameRate.ideal,30);
 assert.equal(requested.audio,false);
 assert.ok(dom('video').srcObject,'camera stream attached');
 h.stop();
});

test('stale camera permission result cannot reattach stream after user stops',async()=>{
 const {h,ctx,dom}=appHarness();let resolveCamera;let closed=0;
 ctx.navigator.mediaDevices.getUserMedia=()=>new Promise(resolve=>{resolveCamera=resolve});
 h.setModelReady(true);
 vm.runInNewContext('initInferenceWorker=async()=>{workerReady=true;return true};toggleEcho=async()=>{};',ctx);
 const pending=h.startCamera();await flush();
 h.stop();resolveCamera({getTracks:()=>[{stop(){closed++}}],getVideoTracks:()=>[]});
 await pending;
 assert.equal(closed,1,'stale media stream is closed');
 assert.equal(dom('video').srcObject,null);
});

test('blocked speech does not spam synth attempts until real user gesture',()=>{
 const {h,speech,windowEvents,clock}=appHarness();h.setLive(true);
 clock.now=1000;assert.equal(h.voiceSay('greeting',clock.now),true);
 speech[0].onerror({error:'not-allowed'});
 for(let t=1100;t<9000;t+=100){clock.now=t;assert.equal(h.voiceSay('greeting',clock.now),false)}
 assert.equal(speech.length,1,'no infinite TTS retry storm');
 windowEvents.pointerdown();clock.now=9500;
 assert.equal(h.voiceSay('greeting',clock.now),true);
 assert.equal(speech.length,2);
});

test('initial greeting must be canceled if tracking is lost before 680ms hold',()=>{
 const {h,speech,clock}=appHarness();h.setLive(true);
 h.applyFaceResult(face(),1000);
 h.applyFaceResult({blendScores:{}},1330);
 h.applyFaceResult(face(),1400);
 clock.now=1770;h.applyFaceResult(face(),1770);
 assert.equal(speech.length,0,'must not greet based on elapsed time from vanished face');
 clock.now=2100;h.applyFaceResult(face(),2100);
 assert.equal(speech.length,1);
});

test('serial avatar switching honors the latest request even during an in-flight VRM bind',async()=>{
 const {h,ctx,radios}=appHarness();
 h.setSquirrel({id:'squirrel'});
 let releaseBind,bindStarted=false,activeModel='squirrel';
 ctx.__testBind=(g,kind)=>{
  if(kind==='anime')return new Promise(resolve=>{bindStarted=true;releaseBind=()=>{activeModel=kind;resolve()}});
  activeModel=kind;
 };
 vm.runInNewContext('bindAvatar=async (g,kind)=>{await __testBind(g,kind);boundAvatarKind=kind};boundAvatarKind="squirrel";',ctx);
 const first=h.switchAvatar('anime');
 await flush();assert.equal(bindStarted,true,'first bind started');
 const latest=h.switchAvatar('squirrel');
 releaseBind();await Promise.all([first,latest]);
 assert.equal(h.getCurrent(),'squirrel');assert.equal(h.getBound(),'squirrel');assert.equal(activeModel,'squirrel','in-flight stale bind restored');
 assert.equal(radios[0].checked,true);assert.equal(radios[1].checked,false);
});

test('avatar choice changed during first download cannot install a stale model',async()=>{
 const {h,ctx}=appHarness();h.setSquirrel({id:'squirrel'});
 let releaseLoad,applied=[];
 const deferred=new Promise(resolve=>{releaseLoad=()=>resolve({id:'anime'})});
 ctx.GLTFLoader=class{loadAsync(){return deferred}};
 ctx.__testBind=async(g,kind)=>{applied.push(kind)};
 vm.runInNewContext('bindAvatar=async(g,kind)=>{await __testBind(g,kind);boundAvatarKind=kind};boundAvatarKind="squirrel";',ctx);
 const first=h.switchAvatar('anime');await flush();
 const latest=h.switchAvatar('squirrel');releaseLoad();
 await Promise.all([first,latest]);
 assert.equal(h.getCurrent(),'squirrel');assert.equal(h.getBound(),'squirrel');
 assert.deepEqual(applied,[],'stale download never changes the displayed model');
});

test('worker 900-frame virtual inference: always closes full 720p frames and transmits used channels only',async()=>{
 const messages=[];let canvasDraws=0,faceDetections=0,poseDetections=0;
 class OffscreenCanvas{constructor(w,h){this.width=w;this.height=h}getContext(){return{drawImage(){canvasDraws++}}}}
 const ctx={OffscreenCanvas,postMessage:data=>messages.push(data),performance:{now:()=>42},console};
 vm.runInNewContext(fs.readFileSync(new URL('../inference-worker.js',import.meta.url),'utf8'),ctx,{timeout:3000});
 ctx.__face=()=>{faceDetections++;return {faceBlendshapes:[{categories:[{categoryName:'jawOpen',score:.9},{categoryName:'mouthSmileLeft',score:.7},{categoryName:'unusedHugeChannel',score:.7}]}],faceLandmarks:[Array.from({length:478},(_,i)=>({x:i/500,y:.5,z:0}))]}};
 ctx.__pose=()=>{poseDetections++;return {landmarks:[Array.from({length:33},(_,i)=>({x:i/33,y:.5,z:.1,visibility:1,presence:1}))]}};
 vm.runInNewContext('face={detectForVideo:__face};pose={detectForVideo:__pose};',ctx);
 for(let i=0;i<900;i++){
  const frame={width:1280,height:720,closed:0,close(){this.closed++}};
  await ctx.onmessage({data:{type:'frame',bitmap:frame,timestamp:i*70,sessionId:3}});
  assert.equal(frame.closed,1,'ImageBitmap closed exactly once');
 }
 assert.equal(canvasDraws,900);assert.equal(faceDetections,600);assert.equal(poseDetections,300);
 assert.equal(messages.length,900);
 const oneFace=messages.find(m=>m.kind==='face');
 assert.deepEqual(Object.keys(oneFace.face.blendScores).sort(),['jawOpen','mouthSmileLeft']);
 assert.equal(Object.keys(oneFace.face.headPoints).length,3);
 assert.equal(Object.keys(messages.find(m=>m.kind==='pose').pose.points).length,8);
 assert.ok(messages.every(m=>m.sessionId===3));
});

test('worker exception releases ImageBitmap and recovers at the next video frame',async()=>{
 const messages=[];
 class OffscreenCanvas{constructor(w,h){this.width=w;this.height=h}getContext(){return{drawImage(){}}}}
 const ctx={OffscreenCanvas,postMessage:data=>messages.push(data),performance:{now:()=>1},console};
 vm.runInNewContext(fs.readFileSync(new URL('../inference-worker.js',import.meta.url),'utf8'),ctx);
 let fail=true;ctx.__detect=()=>{if(fail){fail=false;throw Error('synthetic vision fault')}return {faceBlendshapes:[],faceLandmarks:[]}};
 vm.runInNewContext('face={detectForVideo:__detect}',ctx);
 for(let i=0;i<2;i++){
  const frame={width:1280,height:720,closed:0,close(){this.closed++}};
  await ctx.onmessage({data:{type:'frame',bitmap:frame,timestamp:1000+i*70,sessionId:5}});
  assert.equal(frame.closed,1);
 }
 assert.equal(messages[0].type,'error');assert.equal(messages[1].type,'result');
});

test('mirror invariant holds for both hand sides and model root rotation',()=>{
 const css=fs.readFileSync(new URL('../index.html',import.meta.url),'utf8');
 const src=getSource();
 assert.match(css,/#three\{z-index:1;transform:scaleX\(-1\)/);
 assert.match(css,/object-fit:contain;transform:scaleX\(-1\)/);
 assert.match(src,/character\.rotation\.y=Math\.PI/);
 for(const dx of [-.7,-.1,.1,.7]){
  const vector={};cameraSegmentToModel({x:.5,y:.5,z:0},{x:.5+dx,y:.5,z:0},vector);
  assert.equal(Math.sign(-vector.x),Math.sign(-dx));
 }
});

test('no resolution scaling hidden in worker, fallback, styles or build output',()=>{
 const src=getSource(),worker=fs.readFileSync(new URL('../inference-worker.js',import.meta.url),'utf8');
 assert.match(src,/ideal:1280/);assert.match(src,/ideal:720/);
 assert.doesNotMatch(src+worker,/drawImage\([^\n]+,0,0,\s*(?:320|480|640),\s*(?:240|360|480)\s*\)/);
 assert.doesNotMatch(src+worker,/(?:renderScale|resolutionScale|dynamicResolution|setResolutionScale)/i);
 assert.match(src,/render\.render|renderer\.render/);
 assert.match(worker,/new OffscreenCanvas\(width,height\)/);
});

test('worker scheduler never queues a second video bitmap while a first frame is in flight',async()=>{
 const {h,ctx,dom,clock,timers}=appHarness();const video=dom('video');
 video.readyState=4;video.currentTime=1;h.setLive(true);
 let bitmapCount=0,posted=0;
 ctx.createImageBitmap=async()=>{bitmapCount++;return {close(){}}};
 h.setWorker({postMessage(){posted++}},true);h.beginInference();
 for(let i=0;i<14;i++){
  const [id,timer]=timers.entries().next().value;timers.delete(id);clock.now+=timer.ms;video.currentTime+=1/30;
  timer.fn();await flush();
 }
 assert.equal(bitmapCount,1,'only one ImageBitmap created until the worker acknowledges it');
 assert.equal(posted,1);
 assert.equal(h.getWorkerState().busy,true);
 h.stopInference();
});

test('late rejected old camera bitmap cannot clear the current worker in-flight flag',async()=>{
 const {h,ctx,dom,clock,timers}=appHarness();const video=dom('video');
 video.readyState=4;video.currentTime=1;h.setLive(true);
 let rejectOld,first=true,posts=0;
 ctx.createImageBitmap=()=>{
  if(first){first=false;return new Promise((_,reject)=>{rejectOld=reject})}
  return Promise.resolve({close(){}});
 };
 h.setWorker({postMessage(){posts++}},true);h.beginInference();
 async function tick(){const [id,t]=timers.entries().next().value;timers.delete(id);clock.now+=t.ms;video.currentTime+=1/30;t.fn();await flush()}
 for(let i=0;i<6;i++)await tick();assert.equal(h.getWorkerState().busy,true);
 h.stopInference();h.beginInference();
 for(let i=0;i<10;i++)await tick();
 assert.equal(posts,1,'new camera frame handed to the worker');
 assert.equal(h.getWorkerState().busy,true);
 rejectOld(Error('old bitmap rejected after reconnect'));await flush();
 assert.equal(h.getWorkerState().busy,true,'old request cannot reset a new in-flight lock');
 h.stopInference();
});

test('manual camera stop also stops microphone recording and releases local audio stream',async()=>{
 const {h,clock}=appHarness();let audioStopped=0,audioClosed=0;
 h.echo.enabled=true;h.echo.stream={getTracks:()=>[{stop(){audioStopped++}}]};
 h.echo.ctx={close:async()=>{audioClosed++}};
 h.setLive(true);clock.now=2000;h.stop();await flush();
 assert.equal(h.echo.enabled,false);
 assert.equal(audioStopped,1);assert.equal(audioClosed,1);
});

test('real main draw drives neck/head back for chin-up and forward for chin-down, with full FHD pixels',()=>{
 const {h,dom,clock,ctx}=appHarness();
 const mkBone=()=>({userData:{restRot:{x:0,y:0,z:0}},rotation:{x:0,y:0,z:0,set(x,y,z){this.x=x;this.y=y;this.z=z}}});
 const head=mkBone(),neck=mkBone();h.setHeadRig(head,neck);h.setLive(true);
 const canvas=dom('three');canvas.clientWidth=1920;canvas.clientHeight=1080;
 let ratio=1,rendered=0;h.setRender({setPixelRatio(v){ratio=v},setSize(w,ht){canvas.width=w*ratio;canvas.height=ht*ratio},render(){rendered++}},{});
 const matrix=x=>{const c=Math.cos(x),s=Math.sin(x);return [1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1]};
 const input=angle=>({blendScores:{jawOpen:.01,mouthSmileLeft:.75,mouthSmileRight:.70,eyeBlinkLeft:.8},headMatrix:matrix(angle)});
 clock.now=100;h.applyFaceResult(input(-.45),100);
 assert.ok(h.getTarget().pitch<-.4,'upward tilt detected in raw MediaPipe coordinates');
 for(let i=0;i<90;i++){clock.now=100+i*1000/60;h.draw(clock.now)}
 assert.ok(head.rotation.x>0.2,`head bends backward: ${head.rotation.x}`);
 assert.ok(neck.rotation.x>0.05,`neck also bends backward: ${neck.rotation.x}`);
 assert.ok(head.rotation.x>neck.rotation.x,'head contributes more motion than neck');
 assert.ok(h.getCurrentPose().joy>.3,'facial expression continues while head is tilted');
 assert.ok(h.getCurrentPose().blink_l>.5,'one-eye blink remains active');
 clock.now=2000;h.applyFaceResult(input(.45),2000);
 for(let i=0;i<90;i++){clock.now=2000+i*1000/60;h.draw(clock.now)}
 assert.ok(head.rotation.x<-.2,`head bends forward: ${head.rotation.x}`);
 assert.ok(neck.rotation.x<-.05,`neck bends forward: ${neck.rotation.x}`);
 assert.equal(canvas.width,1920);assert.equal(canvas.height,1080);
 assert.equal(rendered,180);
});

test('3,600 tracked 60Hz frames never accumulate idle pitch/yaw/roll or facial smile into the tracker',()=>{
 const {h,dom,clock}=appHarness();h.setLive(true);
 const canvas=dom('three');canvas.clientWidth=1920;canvas.clientHeight=1080;
 let ratio=1,rendered=0;h.setRender({setPixelRatio(v){ratio=v},setSize(w,ht){canvas.width=w*ratio;canvas.height=ht*ratio},render(){rendered++}},{});
 const a=.35,c=Math.cos(a),s=Math.sin(a),matrix=[1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1];
 clock.now=100;h.applyFaceResult({blendScores:{mouthSmileLeft:.8,mouthSmileRight:.8,jawOpen:.02},headMatrix:matrix},100);
 const original=h.getTarget();
 for(let i=0;i<3600;i++){clock.now=100+i*1000/60;h.draw(clock.now)}
 const next=h.getTarget();
 for(const k of ['pitch','yaw','roll','joy','a'])assert.equal(next[k],original[k],`${k} tracking value must not drift after 60s`);
 assert.equal(rendered,3600);assert.equal(canvas.width,1920);assert.equal(canvas.height,1080);
});

test('mirror-facing avatar head yaw and roll take the expected local bone signs',()=>{
 const {h,dom,clock}=appHarness();h.setLive(true);
 const head={userData:{restRot:{x:0,y:0,z:0}},rotation:{set(x,y,z){this.x=x;this.y=y;this.z=z}}};h.setHeadRig(head,null);
 const c=dom('three');c.clientWidth=1280;c.clientHeight=720;
 h.setRender({setPixelRatio(){},setSize(){},render(){}},{});
 const turn=.4,co=Math.cos(turn),si=Math.sin(turn);
 const yaw=[co,0,-si,0,0,1,0,0,si,0,co,0,0,0,0,1];
 h.applyFaceResult({blendScores:{jawOpen:0},headMatrix:yaw},100);
 for(let i=0;i<50;i++){clock.now=100+i*1000/60;h.draw(clock.now)}
 assert.ok(head.rotation.y>.15,`head yaw should move to unmirrored camera right ${head.rotation.y}`);
 const roll=[co,si,0,0,-si,co,0,0,0,0,1,0,0,0,0,1];
 h.applyFaceResult({blendScores:{jawOpen:0},headMatrix:roll},1100);
 for(let i=0;i<60;i++){clock.now=1100+i*1000/60;h.draw(clock.now)}
 assert.ok(head.rotation.z<-.12,`roll must invert on local head axis after VRM root yaw PI: ${head.rotation.z}`);
});
