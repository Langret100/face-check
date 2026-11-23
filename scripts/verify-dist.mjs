import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..','dist');
const files=[
 ['index.html',5000],['main.js',30000],['interaction-core.js',1000],['inference-worker.js',2000],
 ['assets/mr-squirrel.vrm',1000000],['assets/anime-female.vrm',500000],['assets/squirrel-burrow.webp',10000],
 ['vendor/three/build/three.module.js',100000],
 ['vendor/mediapipe/vision_bundle.mjs',100000],
 ['vendor/mediapipe/wasm/vision_wasm_internal.js',50000],
 ['vendor/mediapipe/wasm/vision_wasm_internal.wasm',1000000],
 ['models/face_landmarker.task',1000000],['models/pose_landmarker_lite.task',1000000]
];
for(const [name,minBytes] of files){
 const location=path.join(root,name);
 if(!fs.existsSync(location))throw Error('GitHub Pages build missing: '+name);
 const bytes=fs.statSync(location).size;
 if(bytes<minBytes)throw Error('GitHub Pages build has truncated asset: '+name+' ('+bytes+' bytes)');
 if(name.endsWith('.wasm')&&!fs.readFileSync(location).subarray(0,4).equals(Buffer.from([0,97,115,109])))
  throw Error('Invalid WASM binary: '+name);
 if(name.endsWith('.vrm')&&fs.readFileSync(location).subarray(0,4).toString()!=='glTF')
  throw Error('Invalid VRM file: '+name);
 console.log('OK',name,bytes);
}
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
if(!html.includes('v=25-loader-repair'))throw Error('HTML is not pointing to the new v25 engine');
console.log('PASS: all deploy-time AI models and runtime binaries are present and nontrivial.');
