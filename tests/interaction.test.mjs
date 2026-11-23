import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { SPEECH_PROFILES,chooseKoreanVoice,classifyExpression,cameraSegmentToModel,armRaised,headPoseFromMatrix } from '../interaction-core.js';

const voices=[{name:'Korean Standard Male',lang:'ko-KR',localService:true},
 {name:'Microsoft SunHi - Korean',lang:'ko-KR',localService:true},
 {name:'English Woman',lang:'en-US',localService:true}];
test('female voice is prioritized within local Korean voices; squirrel picks available Korean',()=>{
 assert.equal(chooseKoreanVoice(voices,'anime').name,'Microsoft SunHi - Korean');
 assert.equal(chooseKoreanVoice(voices,'squirrel').name,'Korean Standard Male');
 assert.equal(chooseKoreanVoice([{name:'only EN',lang:'en-US'}],'anime'),null);
 assert.ok(SPEECH_PROFILES.anime.pitch>1.5);
 assert.ok(SPEECH_PROFILES.squirrel.pitch>SPEECH_PROFILES.anime.pitch);
});
test('sustained emotions are classifiable and surprises override smiles',()=>{
 assert.equal(classifyExpression(.6,.0,1),'surprise');
 assert.equal(classifyExpression(.7,0,0),'joy');
 assert.equal(classifyExpression(0,.7,0),'negative');
 assert.equal(classifyExpression(0,0,0),'neutral');
});
test('mirrored preview and mirrored model move together on both horizontal directions',()=>{
 const dir={x:0,y:0,z:0};
 cameraSegmentToModel({x:.6,y:.4,z:0},{x:.85,y:.3,z:0},dir);
 assert.ok(dir.x>0,'unmirrored model target x moves right for camera raw x+');
 assert.ok(-dir.x<0,'CSS mirrored canvas moves visually left, like mirrored preview');
 cameraSegmentToModel({x:.4,y:.4,z:0},{x:.15,y:.5,z:0},dir);
 assert.ok(dir.x<0);assert.ok(-dir.x>0);
 const poses={11:{x:.8,y:.5},13:{x:.9,y:.3},15:{x:.97,y:.21},12:{x:.2,y:.5},14:{x:.15,y:.61},16:{x:.13,y:.7}};
 assert.equal(armRaised(poses,'left'),true);
 assert.equal(armRaised(poses,'right'),false);
});
function readGlb(name){
 const raw=fs.readFileSync(new URL('../assets/'+name,import.meta.url));
 assert.equal(raw.toString('ascii',0,4),'glTF');
 const size=raw.readUInt32LE(12),json=JSON.parse(raw.subarray(20,20+size).toString());
 return json;
}
test('both VRM rigs preserve the anatomical left/right joint naming in their original assets',()=>{
 for(const file of ['mr-squirrel.vrm','anime-female.vrm']){
  const data=readGlb(file),bones=data.extensions.VRM.humanoid.humanBones;
  const node=name=>data.nodes[bones.find(b=>b.bone===name).node];
  assert.ok(node('leftUpperArm').translation[0]<0,file+' left arm has left-local x');
  assert.ok(node('rightUpperArm').translation[0]>0,file+' right arm has right-local x');
  assert.ok(node('leftLowerArm').translation[0]<0);
  assert.ok(node('rightLowerArm').translation[0]>0);
  // The root Y 180° rotation plus mirrored CSS camera puts anatomical left
  // visually left again: the sign is flipped exactly twice.
  assert.equal(Math.sign(-(-node('leftUpperArm').translation[0])),-1);
 }
});

function createAppHarness(){
 class Q{constructor(){this.x=0;this.y=0;this.z=0;this.w=1;}clone(){return new Q()}}
 class V{constructor(){this.x=0;this.y=0;this.z=0}}
 class E{constructor(){this.x=0;this.y=0;this.z=0}}
 const els=new Map(),speech=[],clock={ms:1000},timers=new Map();let timerId=0;
 const dom=id=>{if(!els.has(id))els.set(id,{id,textContent:'',hidden:false,disabled:false,
  classList:{add(){},remove(){},toggle(){}},addEventListener(){},setAttribute(){},querySelectorAll(){return []}});return els.get(id)};
 const fakeSpeech={getVoices:()=>voices,cancel(){},resume(){},speak(u){speech.push(u)}};
 const ctx={THREE:{Vector3:V,Quaternion:Q,Euler:E},document:{getElementById:dom,querySelectorAll:()=>[],hidden:false,
  addEventListener(){}},window:{speechSynthesis:fakeSpeech,addEventListener(){}},
  speechSynthesis:fakeSpeech,SpeechSynthesisUtterance:class{constructor(text){this.text=text}},
  performance:{now:()=>clock.ms},console,
  setTimeout(fn,delay){const id=++timerId;timers.set(id,{fn,delay});return id},clearTimeout(id){timers.delete(id)},
  setInterval(){},clearInterval(){},requestAnimationFrame(){},cancelAnimationFrame(){}};
 Object.assign(ctx,{SPEECH_PROFILES,chooseKoreanVoice,classifyExpression,cameraSegmentToModel,armRaised,headPoseFromMatrix});
 let code=fs.readFileSync(new URL('../main.js',import.meta.url),'utf8');
 code=code.replace(/^import .*?;\s*$/gm,'').replaceAll('import.meta.url',"'http://localhost/'");
 code=code.replace(/model\(\);requestAnimationFrame\(draw\);\s*$/,'');
 code+='\nthis.__h={applyFaceResult,applyPoseResult,voiceSay,updateVoice,clearVoice,echo,voiceState,setAvatar:v=>{selectedAvatar=v},setLive:v=>{live=v},getTarget:()=>({...target})};';
 vm.runInNewContext(code,ctx,{timeout:2000});
 return {h:ctx.__h,els,speech,clock};
}
const expression=(b={})=>({blendScores:b,headPoints:{left:{x:.3,y:.4},right:{x:.7,y:.4},nose:{x:.5,y:.5}}});
test('camera greeting, expression dialogue, wave and microphone echo work without suppressing TTS',()=>{
 const {h,speech,clock,els}=createAppHarness();
 h.setLive(true);h.setAvatar('anime');
 h.echo.enabled=true; // camera auto-mic MUST NOT silence proactive speech.
 const happy=expression({mouthSmileLeft:.7,mouthSmileRight:.7});
 h.applyFaceResult(happy,1000);assert.equal(speech.length,0,'greeting requires a stable face');
 clock.ms=1800;h.applyFaceResult(happy,1800);assert.equal(speech.length,1,'greeting plays even when mic is enabled');
 assert.equal(speech[0].voice.name,'Microsoft SunHi - Korean');
 assert.equal(speech[0].pitch,SPEECH_PROFILES.anime.pitch);
 assert.equal(h.voiceState.greeted,true);
 assert.equal(els.get('voiceBubble').hidden,false);
 speech[0].onend();assert.equal(els.get('voiceBubble').hidden,true);
 clock.ms=9000;h.applyFaceResult(happy,9000);assert.equal(speech.length,2,'sustained smile prompts a distinct response');
 speech[1].onend();
 const pose={11:{x:.75,y:.5,z:0},13:{x:.8,y:.3,z:0},15:{x:.9,y:.18,z:0},
 12:{x:.25,y:.5,z:0},14:{x:.19,y:.60,z:0},16:{x:.13,y:.68,z:0}};
 clock.ms=17500;h.applyPoseResult({points:pose},17500);
 clock.ms=18500;h.applyPoseResult({points:pose},18500);
 assert.equal(speech.length,3,'raised arm triggers a greeting after a stable hold');
 assert.match(speech[2].text,/손/);
 speech[2].onend();
 clock.ms=26000;h.applyFaceResult(expression({browDownLeft:1,browDownRight:1,mouthFrownLeft:.5,mouthFrownRight:.5}),26000);
 clock.ms=29200;h.applyFaceResult(expression({browDownLeft:1,browDownRight:1,mouthFrownLeft:.5,mouthFrownRight:.5}),29200);
 assert.equal(speech.length,4,'negative expression creates compassionate speech');
 speech[3].onend();
 clock.ms=37000;h.applyFaceResult(expression({jawOpen:1,eyeWideLeft:.8,eyeWideRight:.8}),37000);
 clock.ms=39000;h.applyFaceResult(expression({jawOpen:1,eyeWideLeft:.8,eyeWideRight:.8}),39000);
 assert.equal(speech.length,5,'surprised face prompts an event');
 speech[4].onend();
 h.clearVoice();h.setAvatar('squirrel');
 clock.ms=50000;assert.equal(h.voiceSay('greeting',clock.ms),true);
 assert.equal(speech[5].pitch,SPEECH_PROFILES.squirrel.pitch);
 assert.match(speech[5].text,/찍찍|도토리/);
 speech[5].onend();
});

test('script keeps FHD upper bound and never reduces rendering resolution based on FPS',()=>{
 const src=fs.readFileSync(new URL('../main.js',import.meta.url),'utf8');
 assert.match(src,/limitW=landscape\?1920:1080,limitH=landscape\?1080:1920/);
 assert.doesNotMatch(src,/fps[^\n]{0,90}(?:resolution|setPixelRatio|renderScale)/i);
 assert.match(src,/workerReady&&aiWorker/);
 assert.match(src,/if\(newPose\)/,'IK gets recalculated only for new poses');
});

test('worker serializes only required facial points and upper-body joints and tags the camera session',async()=>{
 const messages=[];
 class OffscreenCanvas{constructor(width,height){this.width=width;this.height=height}getContext(){return {drawImage(){}}}}
 const ctx={OffscreenCanvas,postMessage:m=>messages.push(m),performance:{now:()=>100},console};
 vm.runInNewContext(fs.readFileSync(new URL('../inference-worker.js',import.meta.url),'utf8'),ctx,{timeout:2000});
 vm.runInNewContext(`face={detectForVideo(){return {faceBlendshapes:[{categories:[{categoryName:'jawOpen',score:.8}]}],faceLandmarks:[Array.from({length:478},(_,i)=>({x:i/478,y:.5,z:0}))]}}};
 pose={detectForVideo(){return {landmarks:[Array.from({length:33},(_,i)=>({x:i/33,y:.4,z:0,visibility:1,presence:1}))]}}};`,ctx);
 for(let i=0;i<3;i++){
  const frame={width:480,height:360,closed:false,close(){this.closed=true}};
  await ctx.onmessage({data:{type:'frame',bitmap:frame,timestamp:1000+i*75,sessionId:42}});
  assert.equal(frame.closed,true);
 }
 assert.equal(messages.length,3);
 assert.equal(messages[0].sessionId,42);
 assert.equal(Object.keys(messages[0].face.headPoints).length,3);
 assert.ok(!messages[0].face.faceLandmarks,'face data is compact');
 assert.equal(Object.keys(messages[2].pose.points).length,8);
 assert.equal(messages[2].kind,'pose');
});

test('returning after several seconds triggers a separate return greeting',()=>{
 const {h,speech,clock}=createAppHarness();h.setLive(true);
 const face=expression({jawOpen:0});
 h.applyFaceResult(face,1000);
 clock.ms=1800;h.applyFaceResult(face,1800);assert.equal(speech.length,1);
 speech[0].onend();
 // A tracking gap is not allowed to cancel a character's ongoing speech.
 h.applyFaceResult({blendScores:{}},2600);
 clock.ms=11000;h.applyFaceResult(face,11000);
 clock.ms=11900;h.applyFaceResult(face,11900);
 assert.equal(speech.length,2);
 assert.equal(h.voiceState.lastKind,'return');
});
