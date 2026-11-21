import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const el=id=>document.getElementById(id), canvas=el('three'), video=el('video');
const neutral={a:0,i:0,u:0,e:0,o:0,blink_l:0,blink_r:0,joy:0,angry:0,sorrow:0,fun:0,yaw:0,pitch:0,roll:0};
const target={...neutral},current={...neutral};
const exprTargets={a:11,i:9,u:7,e:10,o:8,blink_l:1,blink_r:0,joy:5,angry:4,sorrow:3,fun:6};
let frameAt=0,lastWidth=0,lastHeight=0,autoStarting=false;
// Independent body tracking, decoupled from face detection so face/TTS still work on pose failure.
let poseLandmarker=null,poseLoading=false,poseLastDetection=0,poseLastVideoTime=-1,poseErrors=0;
let poseGoodAt=-10000,armActivity=0,armTarget=0,bodyDetected=false;
let poseRig=null;
const rigBones={};
const aimVec=new THREE.Vector3(),fromVec=new THREE.Vector3(),parentQuat=new THREE.Quaternion(),aimQuat=new THREE.Quaternion(),worldQuat=new THREE.Quaternion(),boneWorld=new THREE.Vector3(),childWorld=new THREE.Vector3();
const poseInfo={left:[],right:[],waist:0,lean:0};

// Speech is based only on sustained facial expressions, never individual transient frames.
const voiceState={kind:'neutral',since:0,lastAt:-30000,lastKind:'',lastSpoken:'',enabled:true};
const voiceLines={joy:['좋은 일 있나 봐?','웃는 얼굴이 정말 보기 좋아!','오늘 기분이 좋은가 봐!'],negative:['무슨 일 있어? 한번 웃어 볼래?','괜찮아? 다람쥐 친구가 응원할게!','조금 지쳤나 봐. 웃는 얼굴도 보여 줘!'],surprise:['와! 깜짝 놀랐나 봐!','무슨 재미있는 걸 봤어?']};
function voiceSay(kind){
 if(!voiceState.enabled||!('speechSynthesis' in window)||document.hidden)return;
 const lines=voiceLines[kind]||[];
 let line=lines.find(x=>x!==voiceState.lastSpoken)||lines[0]; if(!line)return;
 voiceState.lastSpoken=line; voiceState.lastKind=kind; voiceState.lastAt=performance.now();
 const bubble=el('voiceBubble');bubble.textContent=line;bubble.hidden=false;
 try{speechSynthesis.cancel();const s=new SpeechSynthesisUtterance(line);s.lang='ko-KR';s.rate=1.08;s.pitch=1.28;s.volume=.85;
 const voices=speechSynthesis.getVoices();s.voice=voices.find(v=>v.lang.toLowerCase().startsWith('ko')&&v.localService)||voices.find(v=>v.lang.toLowerCase().startsWith('ko'))||null;
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

let renderer,scene,camera,character,headBone,framing=null,morphMeshes=[],live=false,demo=false,session=0,stream=null,landmarker=null,lastVideoTime=-1,lastDetection=0,errorCount=0,hadFace=false,demoStart=0,temporaryEmotion=null,emotionUntil=0,modelReady=false;
function status(x){el('status').textContent=x}function hint(x){el('hint').textContent=x}function err(x){el('error').textContent=x;el('error').classList.add('on')}function clearErr(){el('error').classList.remove('on')}
function enable(){modelReady=true;for(const k of ['demo','happy','angry','sad'])el(k).disabled=false;el('loading').hidden=true;status('캐릭터 준비 완료');hint('카메라 연결을 시작합니다');startCamera();}
function reset(){Object.assign(target,neutral)}
function base64ToArrayBuffer(b64){const bin=atob(b64),arr=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)arr[i]=bin.charCodeAt(i);return arr.buffer}
function stop(){poseGoodAt=-10000;armTarget=0;bodyDetected=false;el('poseStatus').textContent='몸동작 대기';clearVoice();++session;live=false;demo=false;temporaryEmotion=null;if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}video.pause();video.srcObject=null;el('videoBox').classList.remove('on');el('retry').hidden=false;el('stop').disabled=true;el('demo').classList.remove('active');reset();status('캐릭터 준비 완료');hint('카메라 연결이 종료됐습니다')}
function emotion(type){if(!modelReady)return;if(live)stop();demo=false;el('demo').classList.remove('active');reset();temporaryEmotion=type;emotionUntil=performance.now()+4000;status('표정 미리보기: '+({joy:'웃음',angry:'화남',sorrow:'슬픔'})[type]);hint('실제 모델에 들어 있는 표정입니다')}
for(const [id,key] of [['happy','joy'],['angry','angry'],['sad','sorrow']])el(id).onclick=()=>emotion(key);
el('stop').onclick=()=>{stop();clearErr()};
el('demo').onclick=()=>{if(live)stop();demo=!demo;temporaryEmotion=null;demoStart=performance.now();el('demo').classList.toggle('active',demo);status(demo?'표정 미리보기 중':'캐릭터 준비 완료');hint(demo?'눈·입·감정 표정이 순서대로 움직입니다':'카메라를 기다리고 있습니다');if(!demo)reset()};
el('full').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen()}catch(e){err('전체화면 전환을 지원하지 않습니다.')}};

async function model(){
 try{
  renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.0));renderer.outputColorSpace=THREE.SRGBColorSpace;
  scene=new THREE.Scene();scene.add(new THREE.HemisphereLight(0xffffff,0xa0b0cf,2.0));
  const light=new THREE.DirectionalLight(0xffffff,2.2);light.position.set(-2,5,5);scene.add(light);
  camera=new THREE.OrthographicCamera(-1,1,1,-1,.05,100);camera.position.set(0,1.4,5);camera.lookAt(0,1.22,0);
  const loader=new GLTFLoader();const gltf=window.__MODEL_BASE64__?await loader.parseAsync(base64ToArrayBuffer(window.__MODEL_BASE64__), ''):await loader.loadAsync('./assets/mr-squirrel.vrm');
  character=gltf.scene;scene.add(character);
  character.traverse(o=>{if(o.isMesh&&o.morphTargetInfluences?.length){morphMeshes.push(o);o.frustumCulled=false;}if(o.isSkinnedMesh)o.frustumCulled=false;});
  // Original VRM0 humanoid hierarchy: neck/head indexed by source bone structure.
  const vrmHumanoid=gltf.parser.json.extensions?.VRM?.humanoid?.humanBones||[];
  const headID=vrmHumanoid.find(x=>x.bone==='head')?.node;
  if(headID!==undefined)headBone=await gltf.parser.getDependency('node',headID);
  if(headBone){headBone.userData.restRot=headBone.rotation.clone();}
  // VRM faces -Z by convention, while our camera is at +Z. Turn it toward the visitor.
  character.rotation.y=Math.PI;
  // The supplied model is exported in a T-pose. Rotate the upper arms down into a relaxed pose.
  const humanBones=gltf.parser.json.extensions?.VRM?.humanoid?.humanBones||[];
  async function bone(name){const idx=humanBones.find(x=>x.bone===name)?.node;return idx===undefined?null:await gltf.parser.getDependency('node',idx)}
  const leftArm=await bone('leftUpperArm'),rightArm=await bone('rightUpperArm');
  if(leftArm)leftArm.rotation.z+=1.08;
  if(rightArm)rightArm.rotation.z-=1.08;
  const leftForearm=await bone('leftLowerArm'),rightForearm=await bone('rightLowerArm');
  if(leftForearm)leftForearm.rotation.z+=.12;
  if(rightForearm)rightForearm.rotation.z-=.12;
  // Capture the actual bone transforms AFTER removing the T-pose.
  for(const name of ['leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','leftShoulder','rightShoulder','leftHand','rightHand','chest','upperChest']){
   const b=await bone(name);if(b){rigBones[name]={bone:b,rest:b.quaternion.clone()};}
  }
  character.updateMatrixWorld(true);
  const box=new THREE.Box3().setFromObject(character),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());
  if(size.y<.01)throw Error('모델 크기를 읽을 수 없습니다.');
  // Center the actual head and shoulders rather than the whole body.
  const headPos=new THREE.Vector3();if(headBone)headBone.getWorldPosition(headPos);else headPos.set(center.x,box.min.y+size.y*.79,center.z);
  // Camera framing is based on head/chest, never the full-body bounding-box.
  // On portrait phones we fit the shoulders into the narrower canvas.
  framing={height:size.y,width:size.x,centerX:headPos.x,faceZ:headPos.z,
    focusY:box.min.y+size.y*.735};
  camera.position.set(framing.centerX,framing.focusY,framing.faceZ+5);
  camera.lookAt(framing.centerX,framing.focusY,framing.faceZ);
  configureCamera();
  if(!morphMeshes.length)throw Error('표정 메시를 찾지 못했습니다.');
  enable();
 }catch(e){el('loading').textContent='모델 표시 실패';err('3D 모델 로딩 실패: '+(e.message||e));status('모델 로딩 오류');hint('HTTPS로 배포하거나 localhost 서버에서 실행해 주세요');}
}
function configureCamera(){
 if(!camera||!framing)return;
 const w=Math.max(1,canvas.clientWidth),h=Math.max(1,canvas.clientHeight),aspect=w/h;
 // Desktop: face, shoulders and upper belly. Portrait: prioritize face/shoulder width.
 const isPortrait=aspect<.8;
 const desiredHeight=isPortrait ? framing.height*.79 : framing.height*.61;
 const shoulderWidth=framing.width*(isPortrait?.86:.79);
 const viewHeight=Math.max(desiredHeight,shoulderWidth/Math.max(.25,aspect)*1.12)*(1+armActivity*.85);
 const halfH=viewHeight*.5,halfW=halfH*aspect;
 camera.left=-halfW;camera.right=halfW;camera.top=halfH;camera.bottom=-halfH;
 // Shift framing slightly upward for small portrait screens (head stays dominant).
 const focusY=framing.focusY+(isPortrait?.075:0)*framing.height-armActivity*framing.height*.045;
 camera.position.set(framing.centerX,focusY,framing.faceZ+5);
 camera.lookAt(framing.centerX,focusY,framing.faceZ);
 camera.updateProjectionMatrix();
}
function draw(ts){requestAnimationFrame(draw);if(!renderer||!camera)return;
 if(ts-frameAt<15.5)return;frameAt=ts;
 const w=canvas.clientWidth,h=canvas.clientHeight;
 if(w&&h&&(w!==lastWidth||h!==lastHeight)){lastWidth=w;lastHeight=h;renderer.setSize(w,h,false);configureCamera()}
 if(demo&&!live){const t=(ts-demoStart)*.001;reset();const phase=(Math.floor(t/2.7)%6);const s=Math.sin(t*3.0);if(phase===0){target.joy=.85;target.a=Math.max(0,s)*.45}else if(phase===1){target.angry=.9;target.e=.3}else if(phase===2){target.sorrow=.9;target.o=.3}else if(phase===3){target.fun=.9;target.a=Math.max(0,s)*.95}else if(phase===4){target.i=.85;target.u=.25}else{target.a=.9;target.o=.1}target.blink_l=Math.pow(Math.max(0,Math.sin(t*2.3)),25);target.blink_r=target.blink_l;target.yaw=Math.sin(t*.8)*.17;target.roll=Math.sin(t*.95)*.1}
 if(temporaryEmotion){reset();target[temporaryEmotion]=.92;if(ts>emotionUntil){temporaryEmotion=null;reset();status('캐릭터 준비 완료')}}
 for(const key in target)current[key]+=(target[key]-current[key])*.22;
 // Apply VRM0 morph indices directly: verified against uploaded file's blendShapeGroups.
 for(const mesh of morphMeshes){const a=mesh.morphTargetInfluences;for(let i=0;i<a.length;i++)a[i]=0;
  for(const [key,index] of Object.entries(exprTargets))if(index<a.length)a[index]=Math.max(0,Math.min(1,(a[index]||0)+current[key]));
 }
 if(headBone){const rest=headBone.userData.restRot;headBone.rotation.set(rest.x+current.pitch*.23,rest.y+current.yaw*.30,rest.z-current.roll*.25)}
 updatePoseAnimation(ts);
 renderer.render(scene,camera);
 if(live){detect(ts);detectPose(ts);}

}
function score(map,name){return map[name]||0}function clamp(x,min=0,max=1){return Math.max(min,Math.min(max,x))}
function detect(ts){if(!landmarker||video.readyState<2||video.currentTime===lastVideoTime||ts-lastDetection<75)return;lastVideoTime=video.currentTime;lastDetection=ts;
 try{const r=landmarker.detectForVideo(video,ts);const categories=r.faceBlendshapes?.[0]?.categories||[];if(!categories.length){clearVoice();reset();if(hadFace)hint('얼굴을 카메라에 보여 주세요');hadFace=false;return}
 if(!hadFace){hadFace=true;hint('좋아요! 웃거나 눈을 감거나 입을 벌려 보세요')}const b={};for(const p of categories)b[p.categoryName]=p.score;
 reset();target.blink_l=clamp(score(b,'eyeBlinkLeft')*1.35);target.blink_r=clamp(score(b,'eyeBlinkRight')*1.35);
 target.joy=clamp((score(b,'mouthSmileLeft')+score(b,'mouthSmileRight'))*.85);
 target.angry=clamp((score(b,'browDownLeft')+score(b,'browDownRight'))*.7+(score(b,'mouthFrownLeft')+score(b,'mouthFrownRight'))*.25);
 target.sorrow=clamp((score(b,'mouthFrownLeft')+score(b,'mouthFrownRight'))*.65+score(b,'browInnerUp')*.45);
 target.fun=clamp(score(b,'cheekSquintLeft')+score(b,'cheekSquintRight'))*.4;
 const jaw=clamp(score(b,'jawOpen')*1.6),pucker=score(b,'mouthPucker');
 target.a=jaw*(1-pucker)*.9;target.o=clamp(jaw*.6+pucker*.75);target.i=clamp(score(b,'mouthStretchLeft')+score(b,'mouthStretchRight'))*.45;target.u=clamp(score(b,'mouthFunnel')*.8);target.e=clamp(score(b,'mouthPressLeft')+score(b,'mouthPressRight'))*.3;
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
 if(!poseLandmarker||video.readyState<2||video.currentTime===poseLastVideoTime||ts-poseLastDetection<110)return;
 poseLastVideoTime=video.currentTime;poseLastDetection=ts;
 try{
  const result=poseLandmarker.detectForVideo(video,ts);
  const points=result.landmarks?.[0];
  if(!points){poseInfo.left=[];poseInfo.right=[];armTarget=0;return;}
  const visible=(n)=>points[n]&&(points[n].visibility??1)>.48&&(points[n].presence??1)>.45;
  // Force both limb endpoints visible before applying; avoids huge arm swings at occlusion.
  for(const [side,idx] of [['left',[11,13,15]],['right',[12,14,16]]]){
   poseInfo[side]=idx.every(visible)?idx.map(i=>({x:points[i].x,y:points[i].y,z:points[i].z})):[];
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
 bone.quaternion.slerp(desiredLocal,alpha);
 bone.updateMatrixWorld(true);
}
function segmentDirection(p,q){
 // MediaPipe input is unmirrored, while the visitor sees a mirrored selfie.
 // The avatar faces the viewer (+Z), therefore reverse horizontal camera X.
 const dx=-(q.x-p.x),dy=-(q.y-p.y),dz=clamp((p.z-q.z)*.34,-.32,.32);
 const norm=Math.hypot(dx,dy,dz)||1;
 return [dx/norm,dy/norm,dz/norm];
}
function updatePoseAnimation(ts){
 if(!character)return;
 const active=live && (ts-poseGoodAt<800);
 if(!active){armTarget=0;if(bodyDetected&&ts-poseGoodAt>1300){bodyDetected=false;el('poseStatus').textContent='팔·어깨를 카메라에 보여 주세요'}}
 // Animate zoom even when detected arm becomes temporarily invisible.
 const old=armActivity;armActivity+=(armTarget-armActivity)*.075;
 if(Math.abs(armActivity-old)>.00015)configureCamera();
 for(const key of ['chest','upperChest']){
  const entry=rigBones[key];if(!entry)continue;
  const q=entry.rest.clone();if(active){q.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0,0,(poseInfo.lean+poseInfo.waist)*.26)));}
  entry.bone.quaternion.slerp(q,.12);
 }
 character.updateMatrixWorld(true);
 for(const side of ['left','right']){
  const upper=rigBones[side+'UpperArm'],lower=rigBones[side+'LowerArm'];
  const palm=rigBones[side+'Hand'];const marks=poseInfo[side];
  if(active&&marks.length===3&&upper&&lower&&palm){
   const [shoulder,elbow,wrist]=marks;
   const first=segmentDirection(shoulder,elbow),second=segmentDirection(elbow,wrist);
   // The model faces +Z after rotation and image x is screen x. Child vectors supply anatomical sign.
   aimBone(upper,lower.bone,...first,.18);
   aimBone(lower,palm.bone,...second,.2);
  }else{
   for(const entry of [upper,lower])if(entry)entry.bone.quaternion.slerp(entry.rest,.12);
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
 const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480},frameRate:{ideal:30}},audio:false});if(token!==session){s.getTracks().forEach(t=>t.stop());return}stream=s;video.srcObject=s;await video.play();if(token!==session){s.getTracks().forEach(t=>t.stop());return}el('videoBox').classList.add('on');status('AI 모델 준비 중');hint('얼굴 인식 모델을 불러오는 중입니다');await getLandmarker();if(token!==session)return;
 // Pose is optional and starts in parallel; a failed pose download must not block face tracking.
 getPoseLandmarker().catch(e=>console.warn('Pose unavailable:',e));
 live=true;demo=false;lastVideoTime=-1;lastDetection=0;poseLastVideoTime=-1;poseLastDetection=0;errorCount=0;hadFace=false;status('실시간 표정 인식 중');hint('웃거나 팔을 들어 보세요!');s.getVideoTracks()[0]?.addEventListener('ended',()=>{if(token===session){stop();err('카메라 연결이 종료됐습니다.')}});
 }catch(e){if(token===session){stop();err('카메라 연결 실패: '+(e?.message||e));hint('AI 인식 파일 로딩에 실패했어요. 페이지를 새로 고침해 주세요')}}finally{autoStarting=false}}
el('retry').onclick=startCamera;
el('voice').onclick=()=>{voiceState.enabled=!voiceState.enabled;el('voice').textContent=voiceState.enabled?'🔊 목소리 켜짐':'🔇 목소리 꺼짐';if(!voiceState.enabled)clearVoice();else{try{speechSynthesis.getVoices()}catch(e){}}};
window.addEventListener('pointerdown',()=>{if(voiceState.enabled&&'speechSynthesis' in window){speechSynthesis.resume()}},{passive:true});
window.addEventListener('pagehide',stop);document.addEventListener('visibilitychange',()=>{if(document.hidden&&live)stop();else if(!document.hidden&&modelReady&&!live&&!autoStarting)startCamera()});
model();requestAnimationFrame(draw);
