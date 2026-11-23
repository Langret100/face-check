// MediaPipe CPU worker. No browser DOM access, no network upload.
let face=null,pose=null,frameCount=0,working=false,canvas=null,ctx=null,poseInitStarted=false;
let lastTimestamp=0;
// Only the blendshape channels used by applyFaceResult() cross the worker boundary.
const usedFaceKeys=new Set(['eyeBlinkLeft','eyeBlinkRight','mouthSmileLeft','mouthSmileRight',
 'browDownLeft','browDownRight','mouthFrownLeft','mouthFrownRight','browInnerUp',
 'cheekSquintLeft','cheekSquintRight','jawOpen','mouthPucker','mouthStretchLeft',
 'mouthStretchRight','mouthFunnel','mouthPressLeft','mouthPressRight',
 'eyeWideLeft','eyeWideRight']);
const logError=e=>e?.message||String(e);
async function init(root){
 const lib=await import(new URL('vendor/mediapipe/vision_bundle.mjs',root).href);
 const wasm=await lib.FilesetResolver.forVisionTasks(new URL('vendor/mediapipe/wasm',root).href);
 face=await lib.FaceLandmarker.createFromOptions(wasm,{
  baseOptions:{modelAssetPath:new URL('models/face_landmarker.task',root).href,delegate:'CPU'},
  runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:true,outputFacialTransformationMatrixes:true
 });
 postMessage({type:'ready'});
 // Pose is optional and never blocks face readiness.
 poseInitStarted=true;
 try{pose=await lib.PoseLandmarker.createFromOptions(wasm,{
  baseOptions:{modelAssetPath:new URL('models/pose_landmarker_lite.task',root).href,delegate:'CPU'},
  runningMode:'VIDEO',numPoses:1,minPoseDetectionConfidence:.52,minPosePresenceConfidence:.5,
  minTrackingConfidence:.5,outputSegmentationMasks:false
 });postMessage({type:'pose-ready'})}
 catch(e){postMessage({type:'error',message:'Pose optional: '+logError(e)})}
}
onmessage=async ({data})=>{
 if(data.type==='init'){
  try{await init(data.root)}catch(e){postMessage({type:'error',message:logError(e)})}
  return;
 }
 if(data.type!=='frame')return;
 const frame=data.bitmap;
 let frameClosed=false;
 const closeFrame=()=>{if(!frameClosed){frameClosed=true;frame?.close()}};
 if(!face||working){closeFrame();return}
 working=true;
 try{
  const width=frame.width,height=frame.height;
  if(!canvas||canvas.width!==width||canvas.height!==height){canvas=new OffscreenCanvas(width,height);ctx=canvas.getContext('2d',{alpha:false,willReadFrequently:false})}
  ctx.drawImage(frame,0,0);closeFrame();
  // MediaPipe VIDEO timestamps must increase strictly on a single tracker.
  const timestamp=Math.max(lastTimestamp+1,Math.floor(data.timestamp));lastTimestamp=timestamp;
  const kind=pose&&frameCount++%3===2?'pose':'face';
  const start=performance.now();
  if(kind==='pose'){
   const r=pose.detectForVideo(canvas,timestamp);
   // Only joints actually used by the rig cross the worker boundary.
   const marks=r.landmarks?.[0],points=marks?{}:null;
   if(marks)for(const i of [11,12,13,14,15,16,23,24]){
    const p=marks[i];if(p)points[i]={x:p.x,y:p.y,z:p.z,visibility:p.visibility,presence:p.presence};
   }
   postMessage({type:'result',kind,timestamp,sessionId:data.sessionId,pose:{points},cost:performance.now()-start});
  }else{
   const r=face.detectForVideo(canvas,timestamp);
   // Avoid copying all ~478 face landmarks across threads every inference.
   const categories=r.faceBlendshapes?.[0]?.categories||[],scores={};
   for(const c of categories)if(usedFaceKeys.has(c.categoryName))scores[c.categoryName]=c.score;
   const landmarks=r.faceLandmarks?.[0];
   const p=landmarks?{left:landmarks[33],right:landmarks[263],nose:landmarks[1]}:null;
   const headPoints=p?{left:{x:p.left.x,y:p.left.y},right:{x:p.right.x,y:p.right.y},nose:{x:p.nose.x,y:p.nose.y}}:null;
   // The official MediaPipe face transform is 16 floats. Preserve it unchanged
   // for stable chin-up/down, neck pitch and yaw/roll even at image edges.
   const matrix=r.facialTransformationMatrixes?.[0]?.data;
   const headMatrix=matrix?.length===16?Array.from(matrix):null;
   postMessage({type:'result',kind,timestamp,sessionId:data.sessionId,face:{blendScores:scores,headPoints,headMatrix},cost:performance.now()-start});
  }
 }catch(e){closeFrame();postMessage({type:'error',sessionId:data.sessionId,message:logError(e)})}
 finally{working=false}
};
