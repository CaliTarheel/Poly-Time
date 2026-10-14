const assert = require('node:assert/strict');
const {paths, triangles} = require('../web/native_boundaries.js');
const xyz = (lon, lat) => {
  lon *= Math.PI/180; lat *= Math.PI/180;
  return [Math.cos(lat)*Math.cos(lon), Math.cos(lat)*Math.sin(lon), Math.sin(lat)];
};
const data = (a,b,extra={}) => ({boundary_segments:[{geometry_xyz:[xyz(...a),xyz(...b)],
  normal:[0,1,0],code:2,owner_a:1,owner_b:2,down:1,...extra}]});
let checks=0;
function check(name, fn) {fn(); checks++; console.log('PASS '+name);}
check('dateline segments stay local on both map sides',()=>{
  const lines=paths(data([175,0],[-175,0]));
  assert.equal(lines.length,2);
  for(const line of lines) assert.ok(Math.max(...line.points.map(p=>p[0]))-Math.min(...line.points.map(p=>p[0]))<.04);
});
check('pole crossings split at the pole without a map-wide chord',()=>{
  const lines=paths(data([20,80],[-160,80]));
  assert.equal(lines.length,2);
  for(const line of lines) {
    assert.ok(Math.max(...line.points.map(p=>p[0]))-Math.min(...line.points.map(p=>p[0]))<1e-9);
    assert.ok(line.points.some(p=>Math.abs(p[1])<1e-9));
  }
});
check('vector paths are independent of review raster dimensions',()=>{
  const frame=data([12,30],[33,48]);
  assert.deepEqual(paths({...frame,width:96,height:48}),paths({...frame,width:8192,height:4096}));
});
check('saved polarity places teeth on the overriding side',()=>{
  for(const down of [1,2]) {
    const teeth=triangles(data([0,-30],[0,30],{down}),{width:720,height:360});
    assert.ok(teeth.length>0);
    for(const t of teeth) {
      assert.ok(Math.abs(t[0][0]-360)<1e-7);
      assert.ok(Math.abs(t[1][0]-360)<1e-7);
      assert.ok((t[2][0]-360)*(down===1?1:-1)>0);
    }
  }
});
check('non-subduction and missing polarity never create teeth',()=>{
  for(const extra of [{code:1},{code:3},{code:4},{code:5},{code:6},{down:-1}])
    assert.equal(triangles(data([0,-30],[0,30],extra),{width:720,height:360}).length,0);
});
check('a closing boundary without a trench is drawn under its own code',()=>{
  const lines=paths(data([12,30],[33,48],{code:6,down:-1}));
  assert.equal(lines.length,1);
  assert.equal(lines[0].code,6);
  assert.deepEqual(paths(data([12,30],[33,48],{code:7})),[]);
});
check('invalid and empty history geometry is harmless',()=>{
  assert.deepEqual(paths({boundary_segments:[]}),[]);
  assert.deepEqual(paths(data([0,0],[180,0])),[]);
  assert.deepEqual(paths({boundary_segments:[{code:2,geometry_xyz:[[NaN,0,1],[1,0,0]]}]}),[]);
});
console.log(`${checks} native vector boundary checks passed.`);
