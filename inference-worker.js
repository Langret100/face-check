// MediaPipe CPU worker. No browser DOM access, no network upload.
let face=null,pose=null,frameCount=0,working=false,canvas=null,ctx=null,poseInitStarted=false;
let lastTimestamp=0;
const logError=e=>e?.message||String(e);
async function init(root){
 const lib=await import(new URL('vendor/mediapipe/vision_bundle.mjs',root).href);
 const wasm=await lib.FilesetResolver.forVisionTasks(new URL('vendor/mediapipe/wasm',root).href);
 face=await lib.FaceLandmarker.createFromOptions(wasm,{
  baseOptions:{modelAssetPath:new URL('models/face_landmarker.task',root).href,delegate:'CPU'},
  runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:true
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
 if(!face||working){frame?.close();return}
 working=true;
 try{
  const width=frame.width,height=frame.height;
  if(!canvas||canvas.width!==width||canvas.height!==height){canvas=new OffscreenCanvas(width,height);ctx=canvas.getContext('2d',{alpha:false,willReadFrequently:false})}
  ctx.drawImage(frame,0,0);frame.close();
  // MediaPipe VIDEO timestamps must increase strictly on a single tracker.
  const timestamp=Math.max(lastTimestamp+1,Math.floor(data.timestamp));lastTimestamp=timestamp;
  const kind=pose&&frameCount++%3===2?'pose':'face';
  const start=performance.now();
  if(kind==='pose'){
   const r=pose.detectForVideo(canvas,timestamp);
   postMessage({type:'result',kind,timestamp,pose:{landmarks:r.landmarks},cost:performance.now()-start});
  }else{
   const r=face.detectForVideo(canvas,timestamp);
   postMessage({type:'result',kind,timestamp,face:{faceBlendshapes:r.faceBlendshapes,faceLandmarks:r.faceLandmarks},cost:performance.now()-start});
  }
 }catch(e){frame?.close();postMessage({type:'error',message:logError(e)})}
 finally{working=false}
};
