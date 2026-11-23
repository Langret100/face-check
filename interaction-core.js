// Pure, dependency-free rules shared by the app and its automated tests.
// The webcam pixels are NOT mirrored before MediaPipe processes them.
// Both the preview video and the final 3D canvas are mirrored by CSS.
export const SPEECH_PROFILES = Object.freeze({
 anime: Object.freeze({pitch:1.72,rate:1.07,volume:.96,echoRate:1.30,highpass:210,presence:3500,presenceGain:4}),
 squirrel: Object.freeze({pitch:1.96,rate:1.26,volume:.97,echoRate:1.62,highpass:345,presence:2650,presenceGain:6})
});
export function chooseKoreanVoice(voices,avatar){
 const korean=voices.filter(v=>typeof v.lang==='string'&&v.lang.toLowerCase().startsWith('ko'));
 if(!korean.length)return null;
 const feminine=/(?:female|woman|여성|여자|지현|유나|서연|하은|선희|sunhi|heami|yuna|seo[- ]?yeon|sora|soyeon|jiyoung|ji[- ]?young)/i;
 const local=korean.filter(v=>v.localService);
 if(avatar==='anime')return local.find(v=>feminine.test(v.name))||korean.find(v=>feminine.test(v.name))||local[0]||korean[0];
 return local[0]||korean[0];
}
export function classifyExpression(smile,negative,surprise){
 if(surprise>.7)return 'surprise';
 if(smile>.42)return 'joy';
 if(negative>.55)return 'negative';
 return 'neutral';
}
export function cameraSegmentToModel(p,q,out){
 // Raw webcam +x -> model +x; CSS mirrors the model and camera identically.
 // Never negate x here, or one side will be mirrored twice.
 const dx=q.x-p.x,dy=p.y-q.y,dz=Math.max(-.32,Math.min(.32,(p.z-q.z)*.34));
 const norm=Math.hypot(dx,dy,dz)||1;
 out.x=dx/norm;out.y=dy/norm;out.z=dz/norm;
 return out;
}
export function armRaised(points,side){
 const a=side==='left'?[11,13,15]:[12,14,16];
 if(!a.every(i=>points[i]&&(points[i].visibility??1)>.48&&(points[i].presence??1)>.45))return false;
 const shoulder=points[a[0]],elbow=points[a[1]],wrist=points[a[2]];
 return shoulder.y-wrist.y>.135 && elbow.y-wrist.y>.075;
}

// MediaPipe returns a column-major 4x4 canonical-face-to-camera transform.
// Extract Euler angles in Y-X-Z order (pitch, yaw, roll) without allocating
// Three.js objects or depending on the camera preview's CSS mirror.
// Returned angles are in radians, in the original unmirrored camera space.
export function headPoseFromMatrix(matrix){
 const m=matrix?.data||matrix;
 if(!m||m.length!==16||![0,1,2,4,5,6,8,9,10].every(i=>Number.isFinite(m[i])))return null;
 const sx=Math.hypot(m[0],m[1],m[2]),sy=Math.hypot(m[4],m[5],m[6]),sz=Math.hypot(m[8],m[9],m[10]);
 if(Math.min(sx,sy,sz)<1e-5)return null;
 const bound=(v)=>Math.max(-.83,Math.min(.83,v));
 // THREE.Euler.setFromRotationMatrix(M, 'YXZ') equivalence (unit scale removed).
 const pitch=Math.asin(Math.max(-1,Math.min(1,-m[9]/sz)));
 const yaw=Math.atan2(m[8]/sz,m[10]/sz);
 const roll=Math.atan2(m[1]/sx,m[5]/sy);
 return {pitch:bound(pitch),yaw:bound(yaw),roll:bound(roll)};
}
