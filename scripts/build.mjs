import fs from 'node:fs';
import path from 'node:path';
import https from 'node:https';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(root, 'dist');
const mkdir = p => fs.mkdirSync(p,{recursive:true});
const cp = (a,b) => { mkdir(path.dirname(b)); fs.copyFileSync(a,b); };
const copyTree = (a,b) => { if(!fs.existsSync(a))throw Error('Missing npm package: '+a);fs.cpSync(a,b,{recursive:true});};
const required = [
  ['index.html','index.html'],['main.js','main.js'],['assets/squirrel-burrow.webp','assets/squirrel-burrow.webp'],['assets/mr-squirrel.vrm','assets/mr-squirrel.vrm'],
  ['node_modules/three/build/three.module.js','vendor/three/build/three.module.js'],
  ['node_modules/@mediapipe/tasks-vision/vision_bundle.mjs','vendor/mediapipe/vision_bundle.mjs']
];
if(fs.existsSync(dist))fs.rmSync(dist,{recursive:true,force:true});
mkdir(dist);
for(const [a,b] of required)cp(path.join(root,a),path.join(dist,b));
copyTree(path.join(root,'node_modules/three/examples/jsm'),path.join(dist,'vendor/three/examples/jsm'));
copyTree(path.join(root,'node_modules/@mediapipe/tasks-vision/wasm'),path.join(dist,'vendor/mediapipe/wasm'));
mkdir(path.join(dist,'models'));
function get(url, target, redirects=0) { return new Promise((resolve,reject)=>{
 if(redirects>5)return reject(Error('Too many redirects: '+url));
 const request=https.get(url,{headers:{'User-Agent':'AI-Expression-Friend-Build/1.0'},timeout:90000},res=>{
  if([301,302,303,307,308].includes(res.statusCode)){
   const next = new URL(res.headers.location,url).toString(); res.resume();get(next,target,redirects+1).then(resolve,reject);return;
  }
  if(res.statusCode!==200){res.resume();reject(Error('HTTP '+res.statusCode+' '+url));return;}
  const file=fs.createWriteStream(target);
  res.pipe(file);
  file.on('finish',()=>file.close(()=>resolve()));
  file.on('error',reject);
 });request.on('timeout',()=>request.destroy(Error('Download timed out: '+url)));request.on('error',reject);
}); }
const models = [
 ['face_landmarker.task','https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',1000000],
 ['pose_landmarker_lite.task','https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',1000000]
];
for (const [name,url,minBytes] of models){
 const target=path.join(dist,'models',name);
 let error;
 for(let attempt=1;attempt<=3;attempt++){
  try{await get(url,target);if(fs.statSync(target).size<minBytes)throw Error('Downloaded model is unexpectedly small');error=null;break;}
  catch(e){error=e;if(fs.existsSync(target))fs.rmSync(target);console.warn('Model download attempt '+attempt+' failed: '+e.message);}
 }
 if(error)throw error;
 console.log('Local model ready:',name,fs.statSync(target).size,'bytes');
}
fs.writeFileSync(path.join(dist,'.nojekyll'),'');
const html=fs.readFileSync(path.join(dist,'index.html'),'utf8');
const js=fs.readFileSync(path.join(dist,'main.js'),'utf8');
if (/cdn\.jsdelivr|unpkg\.com|storage\.googleapis/.test(html+js))throw Error('Runtime still refers to remote assets');
const files=fs.readdirSync(path.join(dist,'vendor/mediapipe/wasm'));
if (!files.some(f=>f.endsWith('.wasm')) || !files.some(f=>f.endsWith('.js')))throw Error('MediaPipe WASM or loader missing');
for(const requiredFile of ['index.html','main.js','assets/mr-squirrel.vrm','vendor/mediapipe/vision_bundle.mjs','vendor/three/build/three.module.js','models/face_landmarker.task','models/pose_landmarker_lite.task']){ if(!fs.existsSync(path.join(dist,requiredFile))) throw Error('Missing deployment asset: '+requiredFile); }
console.log('Self-hosted site ready:',dist,'WASM files:',files.length);
