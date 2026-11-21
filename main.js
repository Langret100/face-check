import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const el=id=>document.getElementById(id), canvas=el('three'), video=el('video');
// One Three.js renderer and one MediaPipe tracker are shared by both VRM models.
// The larger human model loads only when requested in settings.
let selectedAvatar='squirrel',animeModelPromise=null,avatarSwitchToken=0;
const settings=el('characterSettings');
function showSettings(v){settings.hidden=!v;el('settingsButton').setAttribute('aria-expanded',String(v))}
el('settingsButton').addEventListener('click',()=>showSettings(settings.hidden));
el('settingsClose').addEventListener('click',()=>showSettings(false));
async function switchAvatar(name){
 if(name===selectedAvatar)return;
 const token=++avatarSwitchToken;
 const msg=el('characterSettingsStatus');
 el('settingsButton').disabled=true;
 try{
  if(name==='anime'){
   msg.textContent='사람형 3D 모델을 처음 불러오는 중…';
   if(!animeModelPromise)animeModelPromise=new GLTFLoader().loadAsync('./assets/anime-female.vrm').catch(e=>{animeModelPromise=null;throw e});
   const gltf=await animeModelPromise;
   if(token!==avatarSwitchToken)return;
   await bindAvatar(gltf,'anime');
   selectedAvatar='anime';
   el('characterLabel').textContent='Anime · AvatarSample A';
   el('licenseNotice').textContent='사람형: VRoid AvatarSample_A · 공식 무료 이용약관 (CC0 아님)';
   msg.textContent='사람형 캐릭터 선택 완료 · 얼굴·팔 추적 공유';
  }else{
   if(!squirrelScene)throw Error('다람쥐 모델이 아직 준비되지 않았습니다.');
   await bindAvatar(squirrelScene,'squirrel');
   selectedAvatar='squirrel';
   el('characterLabel').textContent='Mr Squirrel';
   el('licenseNotice').textContent='모델: Mr Squirrel · CC0';
   msg.textContent='다람쥐 선택 완료';
  }
  clearVoice();voiceState.lastKind='';voiceState.lastAt=-30000;
  status('캐릭터 준비 완료');
 }catch(e){
  // Rebind the previous cached model if the new one did not pass validation.
  try{if(selectedAvatar==='squirrel'&&squirrelScene)await bindAvatar(squirrelScene,'squirrel');}catch(restoreError){console.error('Restore previous avatar failed',restoreError)}
  msg.textContent='캐릭터 전환 실패: '+(e?.message||e);
  document.querySelector(`input[name=avatar][value=${selectedAvatar}]`).checked=true;
  console.error('Avatar selection failed',e);
 }finally{el('settingsButton').disabled=false}
}
document.querySelectorAll('input[name=avatar]').forEach(input=>input.addEventListener('change',()=>{if(input.checked)switchAvatar(input.value)}));
const neutral={a:0,i:0,u:0,e:0,o:0,blink_l:0,blink_r:0,joy:0,angry:0,sorrow:0,fun:0,yaw:0,pitch:0,roll:0};
const target={...neutral},current={...neutral};
const exprTargets={a:11,i:9,u:7,e:10,o:8,blink_l:1,blink_r:0,joy:5,angry:4,sorrow:3,fun:6};
// v20: FHD pixel count remains fixed; optimize computation and compositing instead.
let frameAt=0,lastWidth=0,lastHeight=0,lastDpr=0,autoStarting=false,inferTurn=0;let fpsFrames=0,fpsAt=0;
// Worker performs CPU MediaPipe inference away from requestAnimationFrame.
// A bounded main-thread fallback keeps the app usable on browsers that restrict workers.
const aiPerf={timer:null,token:0,faceMs:75,poseMs:150,faceCost:0,poseCost:0,nextFace:0,nextPose:0};
let aiWorker=null,workerReady=false,workerBusy=false,workerFrameInFlight=false,workerErrors=0;
let workerFrameId=0,workerLastVideoTime=-1,workerLastSend=0,workerInitPromise=null;
function shutdownWorker(){
 if(aiWorker){aiWorker.terminate();aiWorker=null}
 workerReady=false;workerBusy=false;workerFrameInFlight=false;workerInitPromise=null;
}
async function initInferenceWorker(){
 if(workerReady)return true;
 if(workerInitPromise)return workerInitPromise;
 workerInitPromise=new Promise((resolve,reject)=>{
  if(!window.Worker||!window.createImageBitmap){reject(Error('Worker/ImageBitmap unsupported'));return}
  let settled=false;
  const fail=(err)=>{if(settled)return;settled=true;clearTimeout(watchdog);shutdownWorker();reject(err instanceof Error?err:Error(String(err)))};
  const watchdog=setTimeout(()=>fail(Error('Worker AI initialization timeout')),18000);
  const w=new Worker(new URL('./inference-worker.js',import.meta.url),{type:'module'});
  aiWorker=w;
  w.onmessage=({data})=>{
   if(data.type==='ready'){
    if(!settled){settled=true;clearTimeout(watchdog);workerReady=true;el('poseStatus').textContent='팔 AI 준비 중';resolve(true)}
   } else if(data.type==='pose-ready'){
    el('poseStatus').textContent='몸동작 인식 준비 완료';
   } else if(data.type==='result'){
    workerFrameInFlight=false;
    if(!live||document.hidden)return;
    workerErrors=0;
    if(data.face)applyFaceResult(data.face,data.timestamp);
    if(data.pose)applyPoseResult(data.pose,data.timestamp);
    if(data.poseError)el('poseStatus').textContent='팔 인식 오류 · 얼굴은 계속 동작';
    if(data.cost){
     const key=data.kind==='pose'?'poseCost':'faceCost';
     aiPerf[key]=aiPerf[key]*.8+data.cost*.2;
    }
   } else if(data.type==='error'){
    workerFrameInFlight=false;
    if(!settled){fail(Error(data.message||'Worker initialization error'));return}
    // A failed Pose init must not silently leave the second avatar motionless.
    if(String(data.message||'').startsWith('Pose optional:')){
     el('poseStatus').textContent='팔 AI 재연결 중';
     console.warn('Worker pose unavailable; retrying the standard tracker',data.message);
     useMainInferenceFallback();
     return;
    }
    if(++workerErrors>5){console.warn('Worker failed, switching to CPU fallback',data.message);useMainInferenceFallback()}
   }
  };
  w.onerror=e=>{if(!settled)fail(Error(e.message||'Worker unavailable'));else useMainInferenceFallback()};
  w.postMessage({type:'init',root:new URL('.',import.meta.url).href});
 });
 try{return await workerInitPromise}catch(e){workerInitPromise=null;throw e}
}
function stopInference(){aiPerf.token++;if(aiPerf.timer!==null)clearTimeout(aiPerf.timer);aiPerf.timer=null;workerFrameId++;workerFrameInFlight=false}
function beginInference(){
 stopInference();const token=aiPerf.token;aiPerf.nextFace=0;aiPerf.nextPose=0;
 const tick=()=>{
  if(token!==aiPerf.token||!live||document.hidden){aiPerf.timer=null;return}
  const now=performance.now();
  if(workerReady&&aiWorker){
   // Avoid a permanently stuck in-flight flag if the browser drops a frame.
   if(workerFrameInFlight && now-workerLastSend>2500){console.warn('AI worker frame watchdog exceeded');useMainInferenceFallback();return;}
   const cadence=Math.max(45,Math.min(115,aiPerf.faceCost>32?95:aiPerf.faceCost>18?73:52));
   if(!workerFrameInFlight && video.readyState>=2 && video.currentTime!==workerLastVideoTime && now-workerLastSend>=cadence){
    workerFrameInFlight=true;workerLastSend=now;workerLastVideoTime=video.currentTime;
    const seq=workerFrameId;
    createImageBitmap(video).then(frame=>{
     if(token!==aiPerf.token||seq!==workerFrameId||!aiWorker){frame.close();workerFrameInFlight=false;return}
     try{aiWorker.postMessage({type:'frame',bitmap:frame,timestamp:performance.now()},[frame])}
     catch(e){frame.close();workerFrameInFlight=false;console.warn('Worker transfer failed',e);useMainInferenceFallback()}
    }).catch(e=>{workerFrameInFlight=false;if(++workerErrors>5)useMainInferenceFallback()});
   }
  }else{
   // Max one synchronous CPU inference per tick on browsers without worker support.
   const faceDue=landmarker&&now>=aiPerf.nextFace;
   const poseDue=poseLandmarker&&now>=aiPerf.nextPose;
   const choosePose=poseDue&&(!faceDue||(now-aiPerf.nextPose)/aiPerf.poseMs>(now-aiPerf.nextFace)/aiPerf.faceMs);
   if(choosePose){
    const before=performance.now();detectPose(before);const cost=performance.now()-before;
    aiPerf.poseCost=aiPerf.poseCost*.85+cost*.15;
    aiPerf.poseMs=aiPerf.poseCost>35?240:aiPerf.poseCost>20?180:130;
    aiPerf.nextPose=performance.now()+aiPerf.poseMs;
   }else if(faceDue){
    const before=performance.now();detect(before);const cost=performance.now()-before;
    aiPerf.faceCost=aiPerf.faceCost*.85+cost*.15;
    aiPerf.faceMs=aiPerf.faceCost>28?125:aiPerf.faceCost>16?95:70;
    aiPerf.nextFace=performance.now()+aiPerf.faceMs;
   }
  }
  aiPerf.timer=setTimeout(tick,workerReady?(workerFrameInFlight?30:16):24);
 };
 aiPerf.timer=setTimeout(tick,30);
}
let workerFallbackStarting=false;
async function useMainInferenceFallback(){
 if(workerFallbackStarting)return;
 workerFallbackStarting=true;shutdownWorker();stopInference();
 try{
  await getLandmarker();
  if(live){beginInference();getPoseLandmarker().catch(console.warn)}
 }catch(e){err('얼굴 인식 모델을 시작하지 못했습니다: '+(e.message||e))}
 finally{workerFallbackStarting=false}
}
// Independent body tracking, decoupled from face detection so face/TTS still work on pose failure.
let poseLandmarker=null,poseLoading=false,poseLastDetection=0,poseLastVideoTime=-1,poseErrors=0;
let poseGoodAt=-10000,armActivity=0,armTarget=0,bodyDetected=false;
let poseRig=null;
const rigBones={};
const aimVec=new THREE.Vector3(),fromVec=new THREE.Vector3(),parentQuat=new THREE.Quaternion(),aimQuat=new THREE.Quaternion(),worldQuat=new THREE.Quaternion(),boneWorld=new THREE.Vector3(),childWorld=new THREE.Vector3();
const poseInfo={left:[],right:[],waist:0,lean:0};
const poseSmooth={left:[],right:[]};
let poseBlend=0, lastPoseFrame=0,frameDelta=.016;
// Reuse expression bindings and math temporaries; never allocate arrays in every RAF.
let morphTracks=[];
const poseChestQ=new THREE.Quaternion(),poseLeanQ=new THREE.Quaternion(),poseEuler=new THREE.Euler();
const poseShoulderQ=new THREE.Quaternion(),poseShoulderDelta=new THREE.Quaternion();
const aimDeltaQ=new THREE.Quaternion();
const poseSides=['left','right'];
const torsoKeys=['chest','upperChest'];
const expressionKeys=Object.keys(target);
const exprDrivePairs=Object.entries(exprTargets);
const expressionWeights=Object.create(null);
const smoothing=(a,b,k)=>a+(b-a)*k;
// Procedural idle animation works with either VRM without bundled animation clips.
// Face/pose tracking always takes precedence; easing avoids snapping at handoff.
let idleFaceMix=1, idlePoseMix=1;
let idlePhaseShift=0;
const idleMotionQ=new THREE.Quaternion();
const idleMotionDelta=new THREE.Quaternion();
const idleFrame={t:0,blink:0,glance:0,tilt:0,pitch:0,breath:0,sway:0,animal:false};
function idleSample(ts){
 const t=ts*.001+idlePhaseShift;
 const animal=framing?.kind==='squirrel';
 const blinkCycle=(t+(animal?.38:0))%(animal?3.45:4.05);
 const blink=Math.exp(-Math.pow((blinkCycle-.22)/.105,2));
 const glance=Math.sin(t*(animal?.43:.29))*.095+Math.sin(t*.18+.8)*.045;
 Object.assign(idleFrame,{t,blink,glance,tilt:Math.sin(t*.36+1.1)*(animal?.055:.035),
  pitch:Math.sin(t*.58+.5)*(animal?.052:.036),
  breath:Math.sin(t*(animal?2.15:1.55)),
  sway:Math.sin(t*.57+.3),animal});
 return idleFrame;
}

// Low-latency phrase echo: 170ms end-of-speech hold; local temporary memory only.
// Avoid open-speaker live monitor: it causes acoustic feedback in a public exhibit.
// Never send voice data over the network. A tap is required to request permission.
const echo={enabled:false,starting:false,stream:null,ctx:null,analyser:null,data:null,recorder:null,chunks:[],speaking:false,startedAt:0,quietAt:0,playing:false,source:null,raf:0,cooldown:0,serial:0};
async function stopEcho(){
 ++echo.serial; echo.enabled=false;echo.chunks.length=0;echo.data=null;echo.analyser=null;cancelAnimationFrame(echo.raf);if(echo.recorder&&echo.recorder.state!=='inactive'){echo.recorder.onstop=null;try{echo.recorder.stop()}catch{}}
 try{echo.source?.stop()}catch{};if(echo.source){try{echo.source.disconnect()}catch{};echo.source.buffer=null;}echo.source=null;echo.stream?.getTracks().forEach(t=>t.stop());echo.stream=null;
 try{await echo.ctx?.close()}catch{};echo.ctx=null;echo.recorder=null;echo.speaking=false;echo.playing=false;
 const b=el('echo');if(b)b.textContent='🎤 마이크 켜기';
}
async function toggleEcho(){
 if(echo.starting)return;
 if(echo.enabled){await stopEcho();return;}
 if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder){err('이 브라우저에서는 마이크 녹음을 지원하지 않습니다.');return;}
 const token=++echo.serial;echo.starting=true;el('echo').textContent='🎤 마이크 권한 확인 중';
 try{
  const microphone=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
  if(token!==echo.serial){microphone.getTracks().forEach(t=>t.stop());return;}
  const ctx=new (window.AudioContext||window.webkitAudioContext)();await ctx.resume();if(token!==echo.serial){microphone.getTracks().forEach(t=>t.stop());await ctx.close();return;}
  const src=ctx.createMediaStreamSource(microphone),analyser=ctx.createAnalyser();analyser.fftSize=1024;analyser.smoothingTimeConstant=.15;src.connect(analyser);
  Object.assign(echo,{enabled:true,stream:microphone,ctx,analyser,data:new Uint8Array(analyser.fftSize),speaking:false,playing:false,cooldown:performance.now()+500});
  try{speechSynthesis.cancel();el('voiceBubble').hidden=true;}catch{}
  el('echo').textContent='🎤 말 따라하기 끄기';clearErr();echo.raf=requestAnimationFrame(monitorEcho);
  microphone.getAudioTracks()[0].addEventListener('ended',()=>stopEcho());
 }catch(e){await stopEcho();el('echo').textContent='🎤 마이크 켜기';hint('마이크가 차단됐어요. 버튼을 눌러 허용해 주세요.');console.warn('마이크 자동 시작 제한:',e);}
 finally{echo.starting=false;}
}
function monitorEcho(now){
 if(!echo.enabled||!echo.analyser)return;
 echo.raf=requestAnimationFrame(monitorEcho);
 if(echo.playing||now<echo.cooldown)return;
 echo.analyser.getByteTimeDomainData(echo.data);
 let power=0;for(const x of echo.data){const d=(x-128)/128;power+=d*d}
 const rms=Math.sqrt(power/echo.data.length),isTalking=rms>.031;
 if(!echo.speaking){
  if(echo.recorder?.state==='recording'||!isTalking)return;
  echo.speaking=true;echo.startedAt=now;echo.quietAt=0;echo.chunks=[];el('echo').textContent='🔴 목소리 녹음 중';
  try{
   const mime=['audio/webm;codecs=opus','audio/mp4','audio/webm'].find(x=>MediaRecorder.isTypeSupported(x));
   const rec=new MediaRecorder(echo.stream,mime?{mimeType:mime}:undefined);echo.recorder=rec;
   const recordingToken=echo.serial;rec.ondataavailable=e=>{if(e.data.size&&echo.enabled&&recordingToken===echo.serial)echo.chunks.push(e.data)};
   rec.onstop=()=>{echo.recorder=null;if(echo.enabled&&recordingToken===echo.serial)playEcho(rec.mimeType);else echo.chunks.length=0;};rec.start(40);
  }catch(e){echo.speaking=false;err('음성 녹음을 시작하지 못했습니다: '+e.message)}
 }else{
  if(isTalking)echo.quietAt=0;else if(!echo.quietAt)echo.quietAt=now;
  if((echo.quietAt&&now-echo.quietAt>170)||now-echo.startedAt>5000){
   const duration=now-echo.startedAt;echo.speaking=false;echo.cooldown=now+200;el('echo').textContent='🔊 목소리 변조 중';
   if(echo.recorder?.state==='recording'){if(duration<260){echo.recorder.onstop=null;echo.recorder.stop();echo.chunks.length=0;}else echo.recorder.stop()}
  }
 }
}
async function playEcho(mime){
 if(!echo.enabled||!echo.chunks.length||!echo.ctx){echo.chunks.length=0;return;}
 const token=echo.serial;echo.playing=true;
 try{
  const blob=new Blob(echo.chunks,{type:mime||'audio/webm'});echo.chunks=[];
  const buffer=await echo.ctx.decodeAudioData(await blob.arrayBuffer());
  if(!echo.enabled||token!==echo.serial||!echo.ctx||buffer.duration<.25||buffer.duration>7)return;
  if('speechSynthesis' in window){speechSynthesis.cancel();el('voiceBubble').hidden=true;}
  const source=echo.ctx.createBufferSource(),filter=echo.ctx.createBiquadFilter(),gain=echo.ctx.createGain();
  source.buffer=buffer;const animeVoice=selectedAvatar==='anime';
  source.playbackRate.value=animeVoice?1.18:1.55;
  filter.type='highpass';filter.frequency.value=animeVoice?160:340;gain.gain.value=animeVoice?.92:.88;
  const bright=echo.ctx.createBiquadFilter();bright.type='peaking';bright.frequency.value=animeVoice?3200:2450;bright.Q.value=.8;bright.gain.value=animeVoice?2.5:5;source.connect(filter);filter.connect(bright);bright.connect(gain);gain.connect(echo.ctx.destination);echo.source=source;
  source.onended=()=>{try{source.disconnect();filter.disconnect();bright.disconnect();gain.disconnect()}catch{};source.buffer=null;if(token===echo.serial){echo.playing=false;echo.source=null;echo.cooldown=performance.now()+300;el('echo').textContent='🎤 말 따라하기 끄기'}};source.start();
 }catch(e){console.warn('Local voice echo unavailable:',e)}finally{if(token===echo.serial&&!echo.source){echo.playing=false;echo.cooldown=performance.now()+700;el('echo').textContent='🎤 말 따라하기 끄기'}echo.chunks.length=0;}
}


// Speech is based only on sustained facial expressions, never individual transient frames.
const voiceState={kind:'neutral',since:0,lastAt:-30000,lastKind:'',lastSpoken:'',enabled:true};
const voiceLines={joy:['좋은 일 있나 봐?','웃는 얼굴이 정말 보기 좋아!','오늘 기분이 좋은가 봐!'],negative:['무슨 일 있어? 한번 웃어 볼래?','괜찮아? 내가 응원할게!','조금 지쳤나 봐. 웃는 얼굴도 보여 줘!'],surprise:['와! 깜짝 놀랐나 봐!','무슨 재미있는 걸 봤어?']};
function voiceSay(kind){
 if(!voiceState.enabled||echo.enabled||!('speechSynthesis' in window)||document.hidden)return;
 const lines=voiceLines[kind]||[];
 let line=lines.find(x=>x!==voiceState.lastSpoken)||lines[0]; if(!line)return;
 voiceState.lastSpoken=line; voiceState.lastKind=kind; voiceState.lastAt=performance.now();
 const bubble=el('voiceBubble');bubble.textContent=line;bubble.hidden=false;
 try{speechSynthesis.cancel();const s=new SpeechSynthesisUtterance(line);s.lang='ko-KR';s.rate=selectedAvatar==='anime'?1.02:1.2;s.pitch=selectedAvatar==='anime'?1.27:1.65;s.volume=.85;
 const voices=speechSynthesis.getVoices();const korean=voices.filter(v=>v.lang.toLowerCase().startsWith('ko'));
 // Prefer known feminine Korean voices if installed; browser/OS voices vary.
 const feminine=/female|woman|여성|여자|지현|유나|서연|하은|sunhi|heami|yuna|seo-yeon|sora/i;
 s.voice=selectedAvatar==='anime'?(korean.find(v=>feminine.test(v.name))||korean.find(v=>v.localService)||korean[0]||null):(korean.find(v=>v.localService)||korean[0]||null);
 s.onend=()=>{bubble.hidden=true};s.onerror=()=>{bubble.hidden=true};speechSynthesis.speak(s);
 }catch(e){hint('음성 출력이 제한되어 있어요. 화면을 한 번 눌러 주세요.')}
}
function updateVoice(ts, smile, negative, surprise){
 const kind=smile>.42?'joy':negative>.55?'negative':surprise>.7?'surprise':'neutral';
 if(kind!==voiceState.kind){voiceState.kind=kind;voiceState.since=ts;return}
 if(kind==='neutral'||!live||ts-voiceState.since<(kind==='negative'?2700:1700)||ts-voiceState.lastAt<20000)return;
 if(kind===voiceState.lastKind&&ts-voiceState.lastAt<40000)return;
 voiceSay(kind);voiceState.since=ts;
}
function clearVoice(){voiceState.kind='neutral';voiceState.since=0;try{speechSynthesis.cancel()}catch(e){}const b=el('voiceBubble');if(b)b.hidden=true}

let renderer,scene,camera,character,headBone,framing=null,morphMeshes=[],squirrelScene=null,animeMorphBindings=null,live=false,demo=false,session=0,stream=null,landmarker=null,lastVideoTime=-1,lastDetection=0,errorCount=0,hadFace=false,demoStart=0,temporaryEmotion=null,emotionUntil=0,modelReady=false;
function status(x){el('status').textContent=x}function hint(x){el('hint').textContent=x}function err(x){el('error').textContent=x;el('error').classList.add('on')}function clearErr(){el('error').classList.remove('on')}
function enable(){modelReady=true;for(const k of ['demo','happy','angry','sad'])el(k).disabled=false;el('loading').hidden=true;status('캐릭터 준비 완료');hint('카메라 연결을 시작합니다');startCamera();}
function reset(){Object.assign(target,neutral)}
function base64ToArrayBuffer(b64){const bin=atob(b64),arr=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)arr[i]=bin.charCodeAt(i);return arr.buffer}
function stop(){stopInference();poseGoodAt=-10000;armTarget=0;bodyDetected=false;el('poseStatus').textContent='몸동작 대기';clearVoice();++session;live=false;demo=false;temporaryEmotion=null;if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}video.pause();video.srcObject=null;el('videoBox').classList.remove('on');el('retry').hidden=false;el('stop').disabled=true;el('demo').classList.remove('active');reset();status('캐릭터 준비 완료');hint('카메라 연결이 종료됐습니다')}
function emotion(type){if(!modelReady)return;if(live)stop();demo=false;el('demo').classList.remove('active');reset();temporaryEmotion=type;emotionUntil=performance.now()+4000;status('표정 미리보기: '+({joy:'웃음',angry:'화남',sorrow:'슬픔'})[type]);hint('실제 모델에 들어 있는 표정입니다')}
for(const [id,key] of [['happy','joy'],['angry','angry'],['sad','sorrow']])el(id).onclick=()=>emotion(key);
el('stop').onclick=()=>{stop();clearErr()};
el('demo').onclick=()=>{if(live)stop();demo=!demo;temporaryEmotion=null;demoStart=performance.now();el('demo').classList.toggle('active',demo);status(demo?'표정 미리보기 중':'캐릭터 준비 완료');hint(demo?'눈·입·감정 표정이 순서대로 움직입니다':'카메라를 기다리고 있습니다');if(!demo)reset()};
el('full').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen()}catch(e){err('전체화면 전환을 지원하지 않습니다.')}};

async function bindAvatar(gltf,kind){
 if(character)scene.remove(character);
 character=gltf.scene;scene.add(character);idleFaceMix=1;idlePoseMix=1;idlePhaseShift=(kind==='squirrel'?.7:0);poseInfo.left=[];poseInfo.right=[];poseSmooth.left=[];poseSmooth.right=[];poseGoodAt=-10000;poseBlend=0;
 morphMeshes=[];headBone=null;animeMorphBindings=null;morphTracks=[];
 for(const key of Object.keys(rigBones))delete rigBones[key];
 character.traverse(o=>{if(o.isMesh&&o.morphTargetInfluences?.length){morphMeshes.push(o);o.frustumCulled=false;}if(o.isSkinnedMesh)o.frustumCulled=false;});
 const vrmHumanoid=gltf.parser.json.extensions?.VRM?.humanoid?.humanBones||[];
 async function bone(name){const idx=vrmHumanoid.find(x=>x.bone===name)?.node;return idx===undefined?null:await gltf.parser.getDependency('node',idx)}
 headBone=await bone('head');if(headBone)headBone.userData.restRot=headBone.rotation.clone();
 character.rotation.y=Math.PI;
 const leftArm=await bone('leftUpperArm'),rightArm=await bone('rightUpperArm');
 const leftForearm=await bone('leftLowerArm'),rightForearm=await bone('rightLowerArm');
 if(kind==='anime'){
  // A-pose/rest pose: compute anatomically from arm segments, not a fixed rotation
  // that may turn the VRoid A-pose into a T-pose on a different model.
  character.updateMatrixWorld(true);
  const relaxArm=(upper,lower,side)=>{
   if(!upper||!lower)return;
   upper.updateWorldMatrix(true,false);lower.updateWorldMatrix(true,false);
   const upperAt=new THREE.Vector3(),lowerAt=new THREE.Vector3();
   upper.getWorldPosition(upperAt);lower.getWorldPosition(lowerAt);
   const from=lowerAt.sub(upperAt).normalize();
   const outward=Math.sign(from.x)||(side==='left'?1:-1);
   const target=new THREE.Vector3(outward*.23,-.97,.08).normalize();
   const correction=new THREE.Quaternion().setFromUnitVectors(from,target);
   const world=new THREE.Quaternion();upper.getWorldQuaternion(world);
   const parent=new THREE.Quaternion();upper.parent.getWorldQuaternion(parent);
   const local=parent.invert().multiply(correction.multiply(world));
   upper.quaternion.copy(local);upper.updateMatrixWorld(true);
  };
  relaxArm(leftArm,leftForearm,'left');relaxArm(rightArm,rightForearm,'right');
 }else{
  // Preserve the squirrel's existing arm rest pose.
  if(leftArm)leftArm.rotation.z+=1.08;
  if(rightArm)rightArm.rotation.z-=1.08;
  if(leftForearm)leftForearm.rotation.z+=.12;
  if(rightForearm)rightForearm.rotation.z-=.12;
 }
 for(const name of ['leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','leftShoulder','rightShoulder','leftHand','rightHand','chest','upperChest']){
  const b=await bone(name);if(b)rigBones[name]={bone:b,rest:b.quaternion.clone(),maxAngle:name.includes('LowerArm')?1.65:name.includes('UpperArm')?1.30:.35};
 }
 // VRM0 blendShapeMaster binds each preset to an actual mesh/target, rather than
 // relying on the squirrel's hardcoded morph indices.
 if(kind==='anime'){
  const defs=gltf.parser.json.extensions?.VRM?.blendShapeMaster?.blendShapeGroups||[];
  const keys={a:'a',i:'i',u:'u',e:'e',o:'o',blink:'blink',blink_l:'blink_l',blink_r:'blink_r',joy:'joy',angry:'angry',sorrow:'sorrow',fun:'fun'};
  const mapping={};
  for(const group of defs){
   const key=(group.presetName||group.name||'').toLowerCase().replace(/[^a-z]/g,'');
   let target=keys[key];if(!target&&['happy','smile'].includes(key))target='joy';
   if(!target)continue;
   const binds=[];
   for(const item of (group.binds||[])){
    try{const node=await gltf.parser.getDependency('mesh',item.mesh);
     node.traverse(o=>{if(o.morphTargetInfluences&&item.index<o.morphTargetInfluences.length)binds.push({mesh:o,index:item.index,weight:(item.weight??100)/100})});
    }catch(e){console.warn('VRM expression bind skipped',e)}
   }
   mapping[target]=binds;
  }
  animeMorphBindings=mapping;
 }
 // Construct a compact direct-write table once per avatar switch, not every frame.
 const pairs=new Map();
 const register=(mesh,index,key,weight=1)=>{
  if(!mesh?.morphTargetInfluences||index>=mesh.morphTargetInfluences.length||index<0)return;
  let byIndex=pairs.get(mesh);if(!byIndex){byIndex=new Map();pairs.set(mesh,byIndex)}
  let entry=byIndex.get(index);if(!entry){entry={mesh,index,channels:[]};byIndex.set(index,entry)}
  entry.channels.push({key,weight});
 };
 if(kind==='squirrel'){
  for(const mesh of morphMeshes)for(const [key,index] of exprDrivePairs)register(mesh,index,key);
 }else if(animeMorphBindings){
  for(const [key,items] of Object.entries(animeMorphBindings))for(const item of items)register(item.mesh,item.index,key,item.weight);
 }
 morphTracks=[...pairs.values()].flatMap(map=>[...map.values()]);
 character.updateMatrixWorld(true);
 const box=new THREE.Box3().setFromObject(character),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());
 if(size.y<.01)throw Error('모델 크기를 읽을 수 없습니다.');
 const headPos=new THREE.Vector3();if(headBone)headBone.getWorldPosition(headPos);else headPos.set(center.x,box.min.y+size.y*.79,center.z);
 // Center on actual face rather than full-body bounding box (long hair inflates width).
 const neckBone=await bone('neck'),chestBone=await bone('chest');
 const chestPos=new THREE.Vector3();if(chestBone)chestBone.getWorldPosition(chestPos);else chestPos.set(headPos.x,headPos.y-size.y*.22,headPos.z);
 const headToChest=Math.max(.12,headPos.y-chestPos.y);
 framing={height:size.y,width:size.x,centerX:headPos.x,faceZ:headPos.z,headY:headPos.y,chestY:chestPos.y,headToChest,kind,
  // Keep the squirrel exactly at its pre-v18 full-height camera anchor.
  focusY:kind==='squirrel'?box.min.y+size.y*.735:headPos.y-headToChest*.82};
 camera.position.set(framing.centerX,framing.focusY,framing.faceZ+5);
 camera.lookAt(framing.centerX,framing.focusY,framing.faceZ);
 configureCamera();
 if(!morphMeshes.length)throw Error('표정 메시를 찾지 못했습니다.');
}

async function model(){
 try{
  renderer=new THREE.WebGLRenderer({canvas,antialias:false,alpha:true,powerPreference:'high-performance'});
  // Native device detail where possible; cap drawing buffer at FHD (landscape/portrait).
  renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;
  scene=new THREE.Scene();scene.add(new THREE.HemisphereLight(0xffffff,0xa0b0cf,2.0));
  const light=new THREE.DirectionalLight(0xffffff,2.2);light.position.set(-2,5,5);scene.add(light);
  camera=new THREE.OrthographicCamera(-1,1,1,-1,.05,100);camera.position.set(0,1.4,5);camera.lookAt(0,1.22,0);
  const loader=new GLTFLoader();const gltf=window.__MODEL_BASE64__?await loader.parseAsync(base64ToArrayBuffer(window.__MODEL_BASE64__), ''):await loader.loadAsync('./assets/mr-squirrel.vrm');
  squirrelScene=gltf;
  await bindAvatar(gltf,'squirrel');
  enable();
 }catch(e){el('loading').textContent='모델 표시 실패';err('3D 모델 로딩 실패: '+(e.message||e));status('모델 로딩 오류');hint('HTTPS로 배포하거나 localhost 서버에서 실행해 주세요');}
}
function configureCamera(){
 if(!camera||!framing)return;
 const w=Math.max(1,canvas.clientWidth),h=Math.max(1,canvas.clientHeight),aspect=w/h;
 const isPortrait=aspect<.8;
 let viewHeight,focusY;
 if(framing.kind==='squirrel'){
  // Restored exactly from the original v12/v17 squirrel framing, isolated from the human.
  const desiredHeight=isPortrait?framing.height*.79:framing.height*.61;
  const shoulderWidth=framing.width*(isPortrait?.86:.79);
  viewHeight=Math.max(desiredHeight,shoulderWidth/Math.max(.25,aspect)*1.12);
  focusY=framing.focusY+(isPortrait?.075:0)*framing.height;
 }else{
  // Bigger face/upper bust than v18, with a higher camera aim so the avatar
  // appears a little LOWER in the viewport. No full-body sizing or arm-driven zoom.
  // Human-only portrait: about 8% tighter than v20. Raising the aim point
  // shifts the avatar DOWN on screen so the hair is clear of the top edge.
  // The squirrel branch above remains pixel-for-pixel unchanged.
  viewHeight=Math.max(framing.headToChest*(isPortrait?1.94:1.79),framing.height*(isPortrait?.328:.300));
  focusY=framing.focusY+framing.headToChest*(isPortrait?.57:.53);
 }
 const halfH=viewHeight*.5,halfW=halfH*aspect;
 camera.left=-halfW;camera.right=halfW;camera.top=halfH;camera.bottom=-halfH;
 camera.position.set(framing.centerX,focusY,framing.faceZ+5);
 camera.lookAt(framing.centerX,focusY,framing.faceZ);
 camera.updateProjectionMatrix();
}
function draw(ts){requestAnimationFrame(draw);if(document.hidden||!renderer||!camera)return;
 frameAt=ts;fpsFrames++;
 if(ts-fpsAt>1200){const measured=Math.round(fpsFrames*1000/Math.max(1,ts-fpsAt));const fpsEl=el('fps');if(fpsEl)fpsEl.textContent=`${measured} fps`;fpsFrames=0;fpsAt=ts;}
 const w=canvas.clientWidth,h=canvas.clientHeight;
 if(w&&h&&(w!==lastWidth||h!==lastHeight||lastDpr!==window.devicePixelRatio)){
  lastDpr=window.devicePixelRatio;
  lastWidth=w;lastHeight=h;
  const landscape=w>=h,limitW=landscape?1920:1080,limitH=landscape?1080:1920;
  const ratio=Math.min(window.devicePixelRatio||1,2,limitW/w,limitH/h);
  renderer.setPixelRatio(Math.max(.25,ratio));
  renderer.setSize(w,h,false);configureCamera();
 }
 if(demo&&!live){const t=(ts-demoStart)*.001;reset();const phase=(Math.floor(t/2.7)%6);const s=Math.sin(t*3.0);if(phase===0){target.joy=.85;target.a=Math.max(0,s)*.45}else if(phase===1){target.angry=.9;target.e=.3}else if(phase===2){target.sorrow=.9;target.o=.3}else if(phase===3){target.fun=.9;target.a=Math.max(0,s)*.95}else if(phase===4){target.i=.85;target.u=.25}else{target.a=.9;target.o=.1}target.blink_l=Math.pow(Math.max(0,Math.sin(t*2.3)),25);target.blink_r=target.blink_l;target.yaw=Math.sin(t*.8)*.17;target.roll=Math.sin(t*.95)*.1}
 if(temporaryEmotion){reset();target[temporaryEmotion]=.92;if(ts>emotionUntil){temporaryEmotion=null;reset();status('캐릭터 준비 완료')}}
 // Drive expressive, quiet idle motion while face tracking is missing or camera is off.
 // The preview buttons remain exclusive: idle never stomps on a selected emotion.
 const idle=idleSample(ts);
 const idleAllowed=!demo&&!temporaryEmotion;
 const faceTracking=live&&hadFace;
 const idleFaceGoal=idleAllowed&&!faceTracking?1:0;
 const idleStep=1-Math.exp(-frameDelta*2.5);
 idleFaceMix+=(idleFaceGoal-idleFaceMix)*idleStep;
 if(idleAllowed&&!faceTracking)reset();
 if(idleAllowed&&idleFaceMix>.001){
  const f=idleFaceMix;
  target.blink_l=Math.max(target.blink_l,idle.blink*f);
  target.blink_r=Math.max(target.blink_r,idle.blink*f);
  target.yaw+=idle.glance*f;
  target.roll+=idle.tilt*f;
  target.pitch+=idle.pitch*f;
  // A subtle friendly idle expression, not a fixed open-mouth grin.
  target.joy+=f*(idle.animal?.18:.13)*(0.55+.45*Math.sin(idle.t*.4));
  if(idle.animal){target.a+=f*.035*Math.max(0,Math.sin(idle.t*.65));}
 }

 const dt=Math.max(0,Math.min(.05,(ts-lastPoseFrame)/1000||.016));lastPoseFrame=ts;frameDelta=dt;
 // Time-correct smoothing keeps the same feel at 30/60/120Hz.
 for(const key of expressionKeys){const k=key.startsWith('blink')?.62:key==='yaw'||key==='roll'?.33:key==='joy'||key==='angry'||key==='sorrow'||key==='fun'?.23:.42;current[key]+=(target[key]-current[key])*(1-Math.pow(1-k,dt*60))}
 // Direct-write the cached mesh/morph pairs. No per-frame Map/Object.entries or push/clear.
 for(const track of morphTracks){
  let value=0;
  for(const c of track.channels){const v=c.key==='blink'?Math.max(current.blink_l,current.blink_r):(current[c.key]||0);value+=v*c.weight}
  value=clamp(value);
  if(Math.abs(track.mesh.morphTargetInfluences[track.index]-value)>0.0001)track.mesh.morphTargetInfluences[track.index]=value;
 }
 if(headBone){const rest=headBone.userData.restRot;headBone.rotation.set(rest.x+current.pitch*.23,rest.y+current.yaw*.30,rest.z-current.roll*.25)}
 updatePoseAnimation(ts);
 renderer.render(scene,camera);
 // AI inference runs in its own bounded scheduler; never call detectForVideo inside RAF.

}
function score(map,name){return map[name]||0}function clamp(x,min=0,max=1){return Math.max(min,Math.min(max,x))}
function detect(ts){if(!landmarker||video.readyState<2||video.currentTime===lastVideoTime||ts-lastDetection<46)return;lastVideoTime=video.currentTime;lastDetection=ts;
 try{const r=landmarker.detectForVideo(video,ts);applyFaceResult(r,ts);}
 catch(e){if(++errorCount>=10){stop();err('표정 분석이 반복 실패했습니다: '+(e?.message||e))}}
}
function applyFaceResult(r,ts){
 try{const categories=r.faceBlendshapes?.[0]?.categories||[];if(!categories.length){clearVoice();reset();if(hadFace)hint('얼굴을 카메라에 보여 주세요');hadFace=false;return}
 if(!hadFace){hadFace=true;hint('좋아요! 웃거나 눈을 감거나 입을 벌려 보세요')}const b={};for(const p of categories)b[p.categoryName]=p.score;
 reset();target.blink_l=clamp(score(b,'eyeBlinkLeft')*1.35);target.blink_r=clamp(score(b,'eyeBlinkRight')*1.35);
 target.joy=clamp((score(b,'mouthSmileLeft')+score(b,'mouthSmileRight'))*.66);
 target.angry=clamp((score(b,'browDownLeft')+score(b,'browDownRight'))*.43+(score(b,'mouthFrownLeft')+score(b,'mouthFrownRight'))*.18);
 target.sorrow=clamp((score(b,'mouthFrownLeft')+score(b,'mouthFrownRight'))*.43+score(b,'browInnerUp')*.27);
 target.fun=clamp((score(b,'cheekSquintLeft')+score(b,'cheekSquintRight'))*.23);
 const jaw=clamp(score(b,'jawOpen')*1.6),pucker=score(b,'mouthPucker');
 target.a=jaw*(1-pucker)*.72;target.o=clamp(jaw*.6+pucker*.75);target.i=clamp(score(b,'mouthStretchLeft')+score(b,'mouthStretchRight'))*.45;target.u=clamp(score(b,'mouthFunnel')*.8);target.e=clamp(score(b,'mouthPressLeft')+score(b,'mouthPressRight'))*.3;
 // Lower-amplitude emotions prevent exaggerated VRM shapes from fighting the lip-sync.
 const emotional=['joy','angry','sorrow','fun'];
 let peak=Math.max(...emotional.map(k=>target[k]));
 if(peak>0){for(const k of emotional)target[k]=Math.pow(target[k]/peak,1.45)*Math.min(.78,peak*.82)}
 const mouthSum=target.a+target.i+target.u+target.e+target.o;
 if(mouthSum>1){for(const k of ['a','i','u','e','o'])target[k]/=mouthSum}

 const pts=r.faceLandmarks?.[0];if(pts){const le=pts[33],re=pts[263],nose=pts[1],dist=Math.max(.05,Math.abs(re.x-le.x));target.roll=clamp((re.y-le.y)/dist,-.7,.7);target.yaw=clamp((nose.x-(le.x+re.x)*.5)/dist*1.35,-.7,.7);}
 const isSurprise=jaw>.68&&score(b,'eyeWideLeft')+score(b,'eyeWideRight')>.55;
 updateVoice(ts, target.joy, Math.max(target.angry,target.sorrow),isSurprise?1:0);
 errorCount=0;
 }catch(e){if(++errorCount>=10){stop();err('표정 분석이 반복 실패했습니다: '+(e?.message||e))}}
}

// Self-hosted MediaPipe: no runtime jsDelivr, unpkg or Google Storage dependency.
// GitHub Actions downloads official binaries during BUILD, then publishes everything together.
let visionRuntimePromise=null;
async function loadVisionRuntime(){
 if(visionRuntimePromise)return visionRuntimePromise;
 visionRuntimePromise=(async()=>{
  const lib=await import('./vendor/mediapipe/vision_bundle.mjs');
  if(!lib.FilesetResolver||!lib.FaceLandmarker||!lib.PoseLandmarker)throw Error('로컬 AI 라이브러리 API 누락');
  const wasm=await lib.FilesetResolver.forVisionTasks('./vendor/mediapipe/wasm');
  return {lib,wasm};
 })();
 try{return await visionRuntimePromise}catch(e){visionRuntimePromise=null;throw Error('로컬 AI 파일을 불러오지 못했습니다. GitHub Actions 빌드 결과를 확인하세요: '+(e?.message||e))}
}

// MediaPipe Pose supplies 33 landmarks. Shoulder/elbow/wrist are 11/13/15 and 12/14/16.
// All skeletal work happens in memory, never sends frames to a custom server.
async function getPoseLandmarker(){
 if(poseLandmarker)return poseLandmarker;
 if(poseLoading)return null;
 poseLoading=true;
 try{
  const {lib:{PoseLandmarker},wasm}=await loadVisionRuntime();
  poseLandmarker=await PoseLandmarker.createFromOptions(wasm,{baseOptions:{modelAssetPath:'./models/pose_landmarker_lite.task',delegate:'CPU'},runningMode:'VIDEO',numPoses:1,minPoseDetectionConfidence:.52,minPosePresenceConfidence:.5,minTrackingConfidence:.5,outputSegmentationMasks:false});
  poseErrors=0;el('poseStatus').textContent='몸동작 인식 준비 완료';return poseLandmarker;
 }catch(e){console.warn('Body tracking optional:',e);el('poseStatus').textContent='몸동작 모델을 불러오지 못했어요';return null}
 finally{poseLoading=false}
}
function detectPose(ts){
 if(!poseLandmarker||video.readyState<2||video.currentTime===poseLastVideoTime||ts-poseLastDetection<80)return;
 poseLastVideoTime=video.currentTime;poseLastDetection=ts;
 try{
  const result=poseLandmarker.detectForVideo(video,ts);
  applyPoseResult(result,ts);
 }catch(e){if(++poseErrors===8){console.warn("Pose detection disabled after errors",e);poseLandmarker=null;el("poseStatus").textContent="몸동작 추적 중지 · 얼굴은 유지"}}
}
function applyPoseResult(result,ts){
 try{const points=result.landmarks?.[0];
  if(!points){poseInfo.left=[];poseInfo.right=[];poseSmooth.left=[];poseSmooth.right=[];armTarget=0;return;}
  const visible=(n)=>points[n]&&(points[n].visibility??1)>.48&&(points[n].presence??1)>.45;
  // Force both limb endpoints visible before applying; avoids huge arm swings at occlusion.
  for(const [side,idx] of [['left',[11,13,15]],['right',[12,14,16]]]){
   if(idx.every(visible)){
    const raw=idx.map(i=>({x:points[i].x,y:points[i].y,z:points[i].z}));
    const prev=poseSmooth[side];
    // Reject single-frame spikes and use adaptive damped landmark smoothing.
    poseInfo[side]=raw.map((p,i)=>{
      const q=prev[i];if(!q)return {...p};
      const jump=Math.hypot(p.x-q.x,p.y-q.y);
      if(jump>.30)return {...q};
      const k=clamp(.25+jump*2.4,.25,.64);
      return {x:smoothing(q.x,p.x,k),y:smoothing(q.y,p.y,k),z:smoothing(q.z,p.z,k*.65)};
    });
    poseSmooth[side]=poseInfo[side];
   }else{poseInfo[side]=[];poseSmooth[side]=[];}
  }
  if(visible(11)&&visible(12)){
   const shoulderMid=(points[11].x+points[12].x)*.5;
   const hipVisible=visible(23)&&visible(24);
   const hipMid=hipVisible?(points[23].x+points[24].x)*.5:shoulderMid;
   poseInfo.waist=clamp((shoulderMid-hipMid)*2.1,-.25,.25);
   poseInfo.lean=clamp((points[12].y-points[11].y)*1.8,-.27,.27);
  }
  if(poseInfo.left.length||poseInfo.right.length){
   poseGoodAt=ts;bodyDetected=true;el('poseStatus').textContent='얼굴 + 팔·어깨 추적 중';
   let maxPose=0;
   for(const key of ['left','right']){
    const arm=poseInfo[key];if(arm.length){
     const [shoulder,elbow,wrist]=arm;
     const raise=clamp((shoulder.y-wrist.y)*2,0,1);
     const spread=clamp(Math.abs(wrist.x-shoulder.x)*1.35,0,1);
     maxPose=Math.max(maxPose,raise,spread*.85);
    }
   }
   armTarget=maxPose;
  }else{armTarget=0;}
  poseErrors=0;
 }catch(e){if(++poseErrors===8){console.warn('Pose detection disabled after errors:',e);poseLandmarker=null;el('poseStatus').textContent='몸동작 추적 중지 · 얼굴은 유지';}}
}
// Aim a VRM bone's rest segment at a target world-space direction.
// Parent transforms are recomputed in hierarchy order; rest orientation is retained on dropout.
// Testable shoulder/elbow/wrist retargeting. Target vectors use mirrored-screen coordinates.
// Align the currently posed bone-to-child direction to the desired camera-space
// direction in WORLD space, then convert back into the bone's parent local space.
function aimBone(entry,nextBone,dx,dy,dz,alpha){
 if(!entry||!nextBone)return;
 const bone=entry.bone;
 bone.updateWorldMatrix(true,false);nextBone.updateWorldMatrix(true,false);
 bone.getWorldPosition(boneWorld);nextBone.getWorldPosition(childWorld);
 const from=fromVec.subVectors(childWorld,boneWorld);
 const to=aimVec.set(dx,dy,dz);
 if(from.lengthSq()<1e-9||to.lengthSq()<1e-9)return;
 from.normalize();to.normalize();
 aimQuat.setFromUnitVectors(from,to);
 bone.getWorldQuaternion(worldQuat);
 const desiredWorld=aimQuat.multiply(worldQuat);
 bone.parent.getWorldQuaternion(parentQuat);
 const desiredLocal=parentQuat.invert().multiply(desiredWorld);
 // Anatomically conservative angular cone around the relaxed rest pose.
 const maxAngle=entry.maxAngle||1.25;
 aimDeltaQ.copy(entry.rest).invert().multiply(desiredLocal);
 const angle=2*Math.acos(clamp(Math.abs(aimDeltaQ.w),0,1));
 if(angle>maxAngle){desiredLocal.copy(entry.rest).slerp(desiredLocal,maxAngle/angle)}
 bone.quaternion.slerp(desiredLocal,alpha);
 bone.updateMatrixWorld(true);
}
function segmentDirection(p,q){
 // MediaPipe input is unmirrored, while the visitor sees a mirrored selfie.
 // Render canvas is mirrored in CSS to match the mirrored webcam preview.
 const dx=(q.x-p.x),dy=-(q.y-p.y),dz=clamp((p.z-q.z)*.34,-.32,.32);
 const norm=Math.hypot(dx,dy,dz)||1;
 return [dx/norm,dy/norm,dz/norm];
}
function updatePoseAnimation(ts){
 if(!character)return;
 const active=live && (ts-poseGoodAt<800);
 const idle=idleSample(ts);
 const idleGoal=(!demo&&!temporaryEmotion&&!active)?1:0;
 const smoothIdle=1-Math.exp(-frameDelta*2.0);
 idlePoseMix+=(idleGoal-idlePoseMix)*smoothIdle;
 poseBlend=smoothing(poseBlend,active?1:0,active?.24:.11);
 if(!active){armTarget=0;if(bodyDetected&&ts-poseGoodAt>1300){bodyDetected=false;el('poseStatus').textContent='팔·어깨를 카메라에 보여 주세요'}}
 // Camera remains fixed as arms move. No dynamic zoom or recentering.
 if(active)poseLeanQ.setFromEuler(poseEuler.set(0,0,(poseInfo.lean+poseInfo.waist)*.20*poseBlend));
 for(const key of torsoKeys){
  const entry=rigBones[key];if(!entry)continue;
  poseChestQ.copy(entry.rest);
  if(active)poseChestQ.multiply(poseLeanQ);
  // Very slight breathing and upper-body sway during idle.
  if(idlePoseMix>.001){
   poseEuler.set(idle.breath*.009*idlePoseMix,0,idle.sway*.013*idlePoseMix);
   poseChestQ.multiply(idleMotionDelta.setFromEuler(poseEuler));
  }
  entry.bone.quaternion.slerp(poseChestQ,.18);
 }
 character.updateMatrixWorld(true);
 for(const side of poseSides){
  const shoulderEntry=rigBones[side+'Shoulder'];
  const upper=rigBones[side+'UpperArm'],lower=rigBones[side+'LowerArm'];
  const palm=rigBones[side+'Hand'];const marks=poseInfo[side];
  if(active&&marks.length===3&&upper&&lower&&palm){
   const [shoulder,elbow,wrist]=marks;
   if(shoulderEntry){
    const lift=clamp((shoulder.y-elbow.y)*1.3,0,1);
    poseShoulderQ.copy(shoulderEntry.rest).multiply(poseShoulderDelta.setFromEuler(poseEuler.set(0,0,(side==='left'?1:-1)*lift*.12)));
    shoulderEntry.bone.quaternion.slerp(poseShoulderQ,.2*poseBlend);
    shoulderEntry.bone.updateMatrixWorld(true);
   }
   const first=segmentDirection(shoulder,elbow),second=segmentDirection(elbow,wrist);
   // The model faces +Z after rotation and image x is screen x. Child vectors supply anatomical sign.
   aimBone(upper,lower.bone,...first,.32*poseBlend);
   aimBone(lower,palm.bone,...second,.33*poseBlend);
  }else{
   // Return to relaxed arm rest, with gentle asymmetric movement while idle.
   const sign=side==='left'?1:-1;
   const t=idle.t;
   if(shoulderEntry){
    idleMotionQ.copy(shoulderEntry.rest).multiply(idleMotionDelta.setFromEuler(
     poseEuler.set(Math.sin(t*.82+(sign>0?0:1.1))*.018*idlePoseMix,0,sign*.012*idlePoseMix)));
    shoulderEntry.bone.quaternion.slerp(idleMotionQ,.11);
   }
   if(upper){
    idleMotionQ.copy(upper.rest).multiply(idleMotionDelta.setFromEuler(
     poseEuler.set(Math.sin(t*.62+(sign>0?1.1:2.7))*.028*idlePoseMix,
      0,sign*(.021+.016*idle.sway)*idlePoseMix)));
    upper.bone.quaternion.slerp(idleMotionQ,.11);
   }
   if(lower){
    idleMotionQ.copy(lower.rest).multiply(idleMotionDelta.setFromEuler(
     poseEuler.set(Math.sin(t*.74+(sign>0?2:3.5))*.028*idlePoseMix,0,0)));
    lower.bone.quaternion.slerp(idleMotionQ,.11);
   }
  }
  if(upper)upper.bone.updateMatrixWorld(true);
 }
}

async function getLandmarker(){
 if(landmarker)return landmarker;
 const {lib:{FaceLandmarker},wasm}=await loadVisionRuntime();
 landmarker=await FaceLandmarker.createFromOptions(wasm,{
  baseOptions:{modelAssetPath:'./models/face_landmarker.task',delegate:'CPU'},
  runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:true
 });
 return landmarker;
}
async function startCamera(){if(!modelReady||autoStarting||live)return;autoStarting=true;stop();clearErr();const token=++session;el('retry').hidden=true;el('stop').disabled=false;status('카메라 연결 중');hint('웹캠 권한을 허용해 주세요');try{
 if(!navigator.mediaDevices?.getUserMedia)throw Error('카메라는 HTTPS 또는 localhost에서만 사용 가능합니다.');
 // Begin loading AI model while user is answering the camera permission prompt.
 const faceReady=initInferenceWorker().catch(e=>{console.warn('Worker unavailable; using main-thread inference',e);return getLandmarker()});faceReady.catch(()=>{});
 let s;try{s=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'user'},width:{ideal:480},height:{ideal:360},frameRate:{ideal:30}},audio:false})}
 catch(firstError){if(firstError.name==='OverconstrainedError'||firstError.name==='NotFoundError'||firstError.name==='NotReadableError')s=await navigator.mediaDevices.getUserMedia({video:true,audio:false});else throw firstError;}if(token!==session){s.getTracks().forEach(t=>t.stop());return}stream=s;video.srcObject=s;await video.play();if(token!==session){s.getTracks().forEach(t=>t.stop());return}el('videoBox').classList.add('on');if(!echo.enabled&&!echo.starting)toggleEcho();status('AI 모델 준비 중');hint('얼굴 인식 모델을 불러오는 중입니다');await faceReady;if(token!==session)return;
 // Pose is optional and starts in parallel; a failed pose download must not block face tracking.
 if(!workerReady)setTimeout(()=>{if(token===session)getPoseLandmarker().catch(e=>console.warn('Pose unavailable:',e));},500);
 live=true;beginInference();demo=false;lastVideoTime=-1;lastDetection=0;poseLastVideoTime=-1;poseLastDetection=0;errorCount=0;hadFace=false;status('실시간 표정 인식 중');hint('웃거나 팔을 들어 보세요!');s.getVideoTracks()[0]?.addEventListener('ended',()=>{if(token===session){stop();err('카메라 연결이 종료됐습니다.')}});
 }catch(e){if(token===session){stop();const name=e?.name||'';
 const cameraError=['NotFoundError','NotReadableError','NotAllowedError','OverconstrainedError'].includes(name);
 const message=name==='NotFoundError'?'사용 가능한 카메라를 찾지 못했습니다. 카메라 연결을 확인해 주세요.':name==='NotAllowedError'?'카메라 권한이 차단되었습니다. 주소창에서 카메라를 허용해 주세요.':name==='NotReadableError'?'다른 앱에서 카메라를 사용 중일 수 있습니다. 카메라를 닫고 재시도해 주세요.':(e?.message||String(e));
 err((cameraError?'카메라 연결 실패: ':'AI 인식 시작 실패: ')+message);
 hint(cameraError?'카메라를 연결하거나 권한을 허용한 후 재연결해 주세요':'AI 모델 파일을 확인한 후 다시 연결해 주세요');}}finally{autoStarting=false}}
el('retry').onclick=startCamera;
el('voice').onclick=()=>{voiceState.enabled=!voiceState.enabled;el('voice').textContent=voiceState.enabled?'🔊 목소리 켜짐':'🔇 목소리 꺼짐';if(!voiceState.enabled)clearVoice();else{try{speechSynthesis.getVoices()}catch(e){}}};
window.addEventListener('pointerdown',()=>{if(voiceState.enabled&&'speechSynthesis' in window){speechSynthesis.resume()}},{passive:true});
el('echo').onclick=toggleEcho;window.addEventListener('pagehide',()=>{stop();stopEcho()});document.addEventListener('visibilitychange',()=>{if(document.hidden){if(live)stop();if(echo.enabled)stopEcho()}else if(modelReady&&!live&&!autoStarting)startCamera()});
model();requestAnimationFrame(draw);
