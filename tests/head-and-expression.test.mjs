import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {headPoseFromMatrix, armRaised, cameraSegmentToModel} from '../interaction-core.js';

const EPS=.00001;
const identity=[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
const rx=a=>{const c=Math.cos(a),s=Math.sin(a);return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]};
const ry=a=>{const c=Math.cos(a),s=Math.sin(a);return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]};
const rz=a=>{const c=Math.cos(a),s=Math.sin(a);return [c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1]};

const approx=(a,b,msg)=>assert.ok(Math.abs(a-b)<EPS,`${msg}: ${a} vs ${b}`);
test('official column-major face matrix: neutral, chin up, chin down, turning, side tilts',()=>{
 for(const [matrix,k,angle] of [[identity,'pitch',0],[rx(-.5),'pitch',-.5],[rx(.5),'pitch',.5],[ry(.45),'yaw',.45],[ry(-.45),'yaw',-.45],[rz(.4),'roll',.4],[rz(-.4),'roll',-.4]]){
  const pose=headPoseFromMatrix(matrix);
  approx(pose[k],angle,`${k} angular direction`);
 }
 const pose=headPoseFromMatrix(rx(-.5).map(x=>x*3));
 approx(pose.pitch,-.5,'uniform matrix scale must not corrupt pitch');
});
test('3D head output uses bounded safety range and rejects malformed transforms',()=>{
 assert.equal(headPoseFromMatrix(null),null);
 assert.equal(headPoseFromMatrix([1,2,3]),null);
 assert.equal(headPoseFromMatrix(Array(16).fill(0)),null);
 assert.equal(headPoseFromMatrix([NaN,...identity.slice(1)]),null);
 approx(headPoseFromMatrix(rx(-1.4)).pitch,-.83,'pitch capped to safe neck range');
});
test('web worker forwards 16 original matrix values (not only eye/nose points)', async()=>{
 const messages=[];
 class OffscreenCanvas{constructor(w,h){this.width=w;this.height=h}getContext(){return {drawImage(){}}}}
 const ctx={OffscreenCanvas,performance:{now:()=>0},postMessage:d=>messages.push(d),console};
 vm.runInNewContext(fs.readFileSync(new URL('../inference-worker.js',import.meta.url),'utf8'),ctx);
 const transform=rx(-.48);
 vm.runInNewContext(`face={detectForVideo(){return {faceBlendshapes:[{categories:[{categoryName:'jawOpen',score:.5}]}],facialTransformationMatrixes:[{rows:4,columns:4,data:${JSON.stringify(transform)}}],faceLandmarks:[Array.from({length:478},(_,i)=>({x:i/478,y:.5,z:0}))]}}}`,ctx);
 let released=0;
 await ctx.onmessage({data:{type:'frame',timestamp:5000,sessionId:44,bitmap:{width:1280,height:720,close(){released++}}}});
 assert.equal(released,1);
 assert.equal(messages.length,1);
 assert.equal(messages[0].face.headMatrix.length,16);
 assert.deepEqual([...messages[0].face.headMatrix],transform);
 approx(headPoseFromMatrix(messages[0].face.headMatrix).pitch,-.48,'worker->face pose');
});

test('both left and right arm activation follows the unmirrored camera data exactly once',()=>{
 const pts=()=>({11:{x:.70,y:.48,visibility:1,presence:1},13:{x:.8,y:.57,visibility:1,presence:1},15:{x:.88,y:.68,visibility:1,presence:1},12:{x:.3,y:.48,visibility:1,presence:1},14:{x:.2,y:.57,visibility:1,presence:1},16:{x:.12,y:.68,visibility:1,presence:1}});
 let points=pts();
 assert.equal(armRaised(points,'left'),false);assert.equal(armRaised(points,'right'),false);
 points[13].y=.34;points[15].y=.20;
 assert.equal(armRaised(points,'left'),true);assert.equal(armRaised(points,'right'),false);
 points=pts();points[14].y=.30;points[16].y=.19;
 assert.equal(armRaised(points,'left'),false);assert.equal(armRaised(points,'right'),true);
 points[16].visibility=.1;assert.equal(armRaised(points,'right'),false,'occlusion cannot trigger false positive');
 const dir={x:0,y:0,z:0};cameraSegmentToModel(points[12],points[14],dir);
 assert.ok(dir.x<0);assert.ok(-dir.x>0,'CSS mirror maps raw leftward movement visually right');
});
test('actual VRM preset expressions exist in BOTH avatars for smile, sadness, eye wink, and mouth',()=>{
 for(const file of ['mr-squirrel.vrm','anime-female.vrm']){
  const raw=fs.readFileSync(new URL('../assets/'+file,import.meta.url));
  const size=raw.readUInt32LE(12),gltf=JSON.parse(raw.subarray(20,20+size).toString());
  const groups=gltf.extensions.VRM.blendShapeMaster.blendShapeGroups;
  const has=new Map(groups.map(g=>[g.presetName,(g.binds||[]).length]));
  for(const name of ['joy','angry','sorrow','fun','a','i','u','e','o','blink_l','blink_r'])assert.ok(has.get(name)>0,`${file} missing real ${name} morph bind`);
 }
});
test('head/neck and facial matrix options remain wired across worker and CPU fallback',()=>{
 const main=fs.readFileSync(new URL('../main.js',import.meta.url),'utf8');
 const worker=fs.readFileSync(new URL('../inference-worker.js',import.meta.url),'utf8');
 assert.match(main,/outputFacialTransformationMatrixes:true/);
 assert.match(worker,/outputFacialTransformationMatrixes:true/);
 assert.match(main,/target\.pitch=pose\.pitch/);
 assert.match(main,/neckBone\.rotation\.set\(rest\.x-current\.pitch/);
 assert.match(main,/headBone\.rotation\.set\(rest\.x-current\.pitch/);
 assert.match(main,/headPoseFromMatrix\(r\.headMatrix/);
});
