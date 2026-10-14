/* Recorded spherical boundary segments, projected without a pixel-edge staircase. */
(function (root) {
  "use strict";
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const unit = a => { const n = Math.hypot(...a); return a.map(x => x / n); };
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const angle = (a, b) => Math.acos(clamp(dot(a, b), -1, 1));
  const valid = a => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite) && Math.hypot(...a) > .5;
  const project = p => [(Math.atan2(p[1], p[0]) + Math.PI)/(2*Math.PI), .5-Math.asin(clamp(p[2], -1, 1))/Math.PI];
  const blend = (a, b, t) => unit(a.map((x, i) => (1-t)*x+t*b[i]));
  function records(data) {
    return (data?.boundary_segments || []).filter(r => Number.isInteger(r.code) && r.code >= 1 && r.code <= 6 &&
      Array.isArray(r.geometry_xyz) && r.geometry_xyz.length === 2 && r.geometry_xyz.every(valid) &&
      angle(...r.geometry_xyz.map(unit)) < Math.PI-.001);
  }
  function arc(a, b) {
    a = unit(a); b = unit(b);
    const distance = angle(a, b), normal = cross(a, b);
    // A meridian that crosses a pole has two distinct projected longitudes at
    // that same pole. Split there rather than drawing across the polar map row.
    if (Math.abs(normal[2]) < 1e-12 && distance > 1e-10) {
      for (const p of [[0,0,1], [0,0,-1]]) {
        const left = angle(a, p), right = angle(p, b);
        if (left > 1e-9 && right > 1e-9 && Math.abs(left+right-distance) < 1e-8)
          return [...arc(a, p), ...arc(p, b)];
      }
    }
    const count = Math.max(8, Math.ceil(distance / (Math.PI/180))), points = [];
    const la = Math.atan2(a[1], a[0]), lb = Math.atan2(b[1], b[0]);
    for (let i = 0; i <= count; i++) {
      const p = blend(a, b, i/count), xy = project(p);
      if (Math.hypot(p[0], p[1]) < 1e-10) xy[0] = ((i === 0 ? lb : la)+Math.PI)/(2*Math.PI);
      if (points.length) {
        while (xy[0]-points.at(-1)[0] > .5) xy[0]--;
        while (xy[0]-points.at(-1)[0] < -.5) xy[0]++;
      }
      points.push(xy);
    }
    const copies = [];
    for (const shift of [-1,0,1]) {
      if (Math.max(...points.map(p => p[0]))+shift < 0 || Math.min(...points.map(p => p[0]))+shift > 1) continue;
      copies.push(points.map(p => [p[0]+shift, p[1]]));
    }
    return copies;
  }
  function paths(data) {
    return records(data).flatMap(row => arc(...row.geometry_xyz).map(points => ({code: row.code, points})));
  }
  function triangles(data, {width, height, zoom=1, cx=.5, cy=.5, pixelRatio=1}) {
    const result = [], bins = new Map(), spacing = 22*pixelRatio/zoom;
    const size = 8.5*pixelRatio/zoom, half = 5*pixelRatio/zoom;
    const left = width*(cx-.5/zoom), right = width*(cx+.5/zoom);
    const top = height*(cy-.5/zoom), bottom = height*(cy+.5/zoom);
    const accept = (x,y) => {
      const bx=Math.floor(x/spacing), by=Math.floor(y/spacing);
      for (let yy=by-1; yy<=by+1; yy++) for (let xx=bx-1; xx<=bx+1; xx++)
        for (const p of bins.get(`${xx}:${yy}`) || []) if ((x-p[0])**2+(y-p[1])**2 < spacing**2) return false;
      const key=`${bx}:${by}`; if (!bins.has(key)) bins.set(key,[]); bins.get(key).push([x,y]); return true;
    };
    for (const row of records(data)) {
      if (row.code !== 2 || !valid(row.normal) || ![row.owner_a,row.owner_b].includes(row.down)) continue;
      const [a,b] = row.geometry_xyz.map(unit);
      const over = row.normal.map(x => x*(row.down === row.owner_a ? 1 : -1));
      const length = arc(a,b).reduce((best,line) => Math.max(best, line.slice(1).reduce((s,p,i) =>
        s+Math.hypot((p[0]-line[i][0])*width,(p[1]-line[i][1])*height),0)),0);
      const count = Math.min(512,Math.max(1,Math.ceil(length/spacing)));
      for (let j=0;j<count;j++) {
        const p=blend(a,b,(j+.5)/count), point=project(p);
        const tangent=unit(cross(p,over));
        const along=project(unit(p.map((v,k)=>v+1e-5*tangent[k])));
        const side=project(unit(p.map((v,k)=>v+1e-5*over[k])));
        const delta=q=>{let dx=q[0]-point[0]; if(dx>.5)dx--;if(dx<-.5)dx++;return [dx*width,(q[1]-point[1])*height];};
        const t=delta(along), d=delta(side), norm=Math.hypot(...t);
        if (!(norm>1e-12)) continue;
        t[0]/=norm;t[1]/=norm;
        const sign=(-t[1]*d[0]+t[0]*d[1])>=0 ? 1:-1, n=[-t[1]*sign,t[0]*sign];
        for (const shift of [-1,0,1]) {
          const x=(point[0]+shift)*width,y=point[1]*height;
          if(x<left-size||x>right+size||y<top-size||y>bottom+size||!accept(x,y))continue;
          result.push([[x-half*t[0],y-half*t[1]],[x+half*t[0],y+half*t[1]],[x+size*n[0],y+size*n[1]]]);
        }
      }
    }
    return result;
  }
  const api={paths,triangles};
  if(typeof module!=="undefined" && module.exports)module.exports=api;else root.DeepTimeNativeBoundaries=api;
})(globalThis);
