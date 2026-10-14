/* Motion visibility and projected direction journeys; no server or model runs. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const globe = require('../web/globe.js');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const near = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const xyz = (lon, lat) => {
  const l = lon * Math.PI / 180, a = lat * Math.PI / 180;
  return [Math.cos(a)*Math.cos(l), Math.cos(a)*Math.sin(l), Math.sin(a)];
};
function rotated(point, omega, dt) {
  const magnitude = Math.hypot(...omega), axis = omega.map(v => v / magnitude);
  const angle = magnitude * dt, tangent = cross(axis, point), parallel = dot(axis, point);
  const p = point.map((v, i) => v*Math.cos(angle) + tangent[i]*Math.sin(angle) + axis[i]*parallel*(1-Math.cos(angle)));
  return {lon: Math.atan2(p[1], p[0])*180/Math.PI, lat: Math.asin(Math.max(-1, Math.min(1, p[2])))*180/Math.PI};
}

function context() {
  let matrix = [1, 0, 0, 1, 0, 0], points = [];
  const stack = [];
  return {
    strokes: [], visibleStrokes: [], ellipses: [], textureReads: [], paints: 0,
    get matrix() { return matrix.slice(); }, get stackDepth() { return stack.length; },
    save() { stack.push({matrix: matrix.slice(), strokeStyle: this.strokeStyle, lineWidth: this.lineWidth}); },
    restore() { assert.ok(stack.length, 'Balanced canvas save/restore'); const old = stack.pop(); matrix = old.matrix; this.strokeStyle = old.strokeStyle; this.lineWidth = old.lineWidth; },
    setTransform(...m) { matrix = m; },
    translate(x, y) { matrix[4] += matrix[0]*x + matrix[2]*y; matrix[5] += matrix[1]*x + matrix[3]*y; },
    scale(x, y) { matrix[0] *= x; matrix[1] *= x; matrix[2] *= y; matrix[3] *= y; },
    beginPath() { points = []; }, moveTo(x, y) { points.push([x, y]); }, lineTo(x, y) { points.push([x, y]); },
    ellipse(...args) { this.ellipses.push(args); }, clip() {}, setLineDash() {},
    stroke() {
      const record = {points: points.map(p => p.slice()), matrix: matrix.slice(), style: this.strokeStyle, width: this.lineWidth};
      this.strokes.push(record); this.visibleStrokes.push(record);
    },
    createImageData(width, height) { return {width, height, data: new Uint8ClampedArray(width*height*4)}; },
    getImageData(x, y, width, height) { this.textureReads.push(this.visibleStrokes.length); return this.createImageData(width, height); },
    putImageData() { this.paints++; this.visibleStrokes = []; },
    drawImage() { this.paints++; this.visibleStrokes = []; },
    fillRect() { this.paints++; this.visibleStrokes = []; },
  };
}

const nodes = new Map(), animationFrames = [];
let document;
function element(tag = 'div') {
  const ctx = context(), listeners = new Map();
  const e = {
    tagName: tag.toUpperCase(), width: 600, height: 300,
    rect: {left: 0, top: 0, width: 600, height: 300},
    textContent: '', innerHTML: '', value: '', hidden: false, checked: false,
    disabled: false, dataset: {}, style: {}, children: [], options: [],
    classList: {toggle() {}, add() {}, remove() {}},
    setAttribute() {}, removeAttribute() {}, querySelectorAll: () => [],
    addEventListener(type, fn) { listeners.set(type, [...(listeners.get(type) || []), fn]); },
    removeEventListener(type) { listeners.delete(type); },
    emit(type) { for (const fn of listeners.get(type) || []) fn({target: e}); },
    getContext(type) { return type === '2d' ? ctx : null; },
    getBoundingClientRect() { return this.rect; },
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    get ownerDocument() { return document; },
  };
  return e;
}
for (const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
  const [, tag, attributes, id] = match, e = element(tag);
  e.value = attributes.match(/\bvalue="([^"]*)"/)?.[1] || '';
  // Other overlays stay off so motion strokes can be measured independently.
  nodes.set(id, e);
}
const get = id => { assert.ok(nodes.has(id), `HTML control exists: ${id}`); return nodes.get(id); };
document = {getElementById: get, createElement: element, querySelectorAll: () => [], addEventListener() {}, body: element()};
const sandbox = {
  URLSearchParams, document, location: {search: ''}, console, DeepTimeGlobe: globe,
  window: {addEventListener() {}, devicePixelRatio: 1},
  requestAnimationFrame(fn) { animationFrames.push(fn); },
  setTimeout() {}, clearTimeout() {}, setInterval() {},
  fetch() { throw new Error('Motion rendering must not request simulation data'); },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root, 'web/app.js'), 'utf8').replace('  initialize();',
  '  globalThis.app = {state, mapView, globeView, render, drawGlobe, motionArrowLength, motionAt, drawMotion, updateMotionLegend};'), sandbox);
const app = sandbox.app, mapCanvas = get('world-map'), globeCanvas = get('world-globe');
const mapCtx = mapCanvas.getContext('2d'), globeCtx = globeCanvas.getContext('2d');
let checks = 0;
function check(name, fn) { fn(); checks++; console.log('PASS ' + name); }
function frame(omega = [0, 0, .0002]) {
  const width = 72, height = 36, count = width*height;
  return {width, height, index: 0, time_myr: 172, plates: [{id: 1, angular_velocity: omega}],
    plate: new Int32Array(count).fill(1), crust: new Uint8Array(count),
    elevation: new Float32Array(count).fill(-1000), age: new Float32Array(count)};
}
function useFrame(data) {
  app.state.mode = 'history'; app.state.frame = data; app.state.frames = [{time_myr: data.time_myr}];
  app.state.currentIndex = 0; app.state.layer = 'crust'; app.state.inspection = null;
  app.globeView.textureFrame = data;
}
function flushRender() { assert.equal(animationFrames.length, 1); animationFrames.shift()(); }
function arrows(ctx, start = 0) { return ctx.strokes.slice(start).filter(row => row.style === '#ecf4ed'); }
function shaft(row) {
  const [a, b] = row.points;
  return {x: (a[0]+b[0])/2, y: (a[1]+b[1])/2, dx: b[0]-a[0], dy: b[1]-a[1], length: Math.hypot(b[0]-a[0], b[1]-a[1])};
}
function resetFlat() {
  app.globeView.active = false;
  Object.assign(app.mapView, {zoom: 1, cx: .5, cy: .5});
  mapCanvas.width = 600; mapCanvas.height = 300;
  mapCanvas.rect = {left: 31, top: 47, width: 600, height: 300};
}

check('slow plates remain visible while zero, missing and invalid rotation draw nothing', () => {
  resetFlat();
  const data = frame([0, 0, .0001]), before = JSON.stringify(data.plates), start = mapCtx.strokes.length;
  app.drawMotion(data);
  const drawn = arrows(mapCtx, start);
  assert.equal(drawn.length, 50);
  assert.ok(drawn.every(row => shaft(row).length >= 10 && shaft(row).length < 15));
  assert.equal(JSON.stringify(data.plates), before, 'Rendering preserves recorded rotation');
  for (const omega of [[0, 0, 0], [NaN, 0, 0], [Infinity, 0, 0], [1, 2], undefined]) {
    const invalid = frame(); invalid.plates[0].angular_velocity = omega;
    const first = mapCtx.strokes.length; app.drawMotion(invalid);
    assert.equal(mapCtx.strokes.length, first);
  }
  const absent = frame(); absent.plates = [];
  const missingStart = mapCtx.strokes.length; app.drawMotion(absent);
  assert.equal(mapCtx.strokes.length, missingStart);
  near(app.motionAt(frame([0, 0, 1/637.1]), new Map([[1, [0, 0, 1/637.1]]]), .5, .5).speed, 1);
  assert.equal(mapCtx.stackDepth, 0);
});

check('readable lengths preserve speed ordering and cap extreme fragment motion', () => {
  const speeds = [.01, .1, .5, 1, 5, 30];
  const lengths = speeds.map(app.motionArrowLength);
  assert.ok(lengths.every((v, i) => v > 10 && (i === 0 || v > lengths[i-1])));
  assert.ok(lengths.at(-1) < 50);
  near(app.motionArrowLength(30), app.motionArrowLength(300));
  for (const value of [0, -1, NaN, Infinity]) assert.equal(app.motionArrowLength(value), 0);
});

check('flat wrapping samples both seam owners without reversing arrows or double drawing', () => {
  resetFlat(); const data = frame();
  data.plates.push({id: 2, angular_velocity: [0, 0, -.0002]});
  for (let y = 0; y < data.height; y++) for (let x = data.width/2; x < data.width; x++) data.plate[y*data.width+x] = 2;
  app.mapView.cx = 1;
  useFrame(data); get('show-motion').checked = true;
  const start = mapCtx.strokes.length; app.render(); flushRender();
  const drawn = arrows(mapCtx, start);
  assert.equal(drawn.length, 50, 'One screen grid, not one repeated grid per map copy');
  for (const row of drawn) { const s = shaft(row); assert.ok(s.x < 300 ? s.dx < 0 : s.dx > 0); near(s.dy, 0); }
  const plates = new Map(data.plates.map(p => [p.id, p.angular_velocity]));
  near(app.motionAt(data, plates, -.05, .5).east, app.motionAt(data, plates, .95, .5).east);
  near(app.motionAt(data, plates, 1.05, .5).east, app.motionAt(data, plates, .05, .5).east);
});

check('flat arrow size and density stay constant under zoom and device pixel ratio', () => {
  resetFlat(); const data = frame([0, .0005, 0]);
  // The center sample remains the same physical point at each zoom.
  mapCanvas.rect.width = 540; mapCanvas.rect.height = 300;
  let expected;
  for (const zoom of [1, 4, 16]) for (const dpr of [1, 2]) {
    app.mapView.zoom = zoom; mapCanvas.width = 540*dpr; mapCanvas.height = 300*dpr;
    const start = mapCtx.strokes.length; app.drawMotion(data); const drawn = arrows(mapCtx, start);
    assert.equal(drawn.length, 45);
    const center = drawn.map(shaft).find(p => Math.abs(p.x-270) < 1e-8 && Math.abs(p.y-150) < 1e-8);
    assert.ok(center); expected ??= center.length; near(center.length, expected);
    for (const row of drawn) { near(row.matrix[0], dpr); near(row.matrix[3], dpr); }
  }
  resetFlat();
});

check('north and south globe arrows match independently rotated point projections with offset and DPR', () => {
  const data = frame([.0003, -.0002, .0001]); useFrame(data);
  const priorDpr = globalThis.devicePixelRatio;
  globeCanvas.rect = {left: 73, top: 119, width: 360, height: 300};
  const renderer = globe.create(globeCanvas); app.globeView.renderer = renderer;
  const texture = element('canvas'); texture.width = 2; texture.height = 1; renderer.setTexture(texture);
  try {
    for (const dpr of [1, 3]) for (const lat of [90, -90]) for (const zoom of [1, 4]) {
      globalThis.devicePixelRatio = dpr;
      app.globeView.pose = {lon: 137, lat, zoom}; renderer.draw(app.globeView.pose);
      const start = globeCtx.strokes.length; app.drawMotion(data, true); const drawn = arrows(globeCtx, start);
      assert.ok(drawn.length > 5);
      for (const row of drawn) {
        const s = shaft(row), point = renderer.pick(73+s.x, 119+s.y, app.globeView.pose);
        assert.ok(point, 'Arrow center stays on visible hemisphere');
        const a = rotated(xyz(point.lon, point.lat), data.plates[0].angular_velocity, -.001);
        const b = rotated(xyz(point.lon, point.lat), data.plates[0].angular_velocity, .001);
        const pa = globe.project(a.lon, a.lat, app.globeView.pose), pb = globe.project(b.lon, b.lat, app.globeView.pose);
        const dx = pb.x-pa.x, dy = pa.y-pb.y, norm = Math.hypot(dx, dy);
        near((s.dx*dy-s.dy*dx)/(s.length*norm), 0, 5e-8);
        assert.ok(s.dx*dx+s.dy*dy > 0, 'Arrow points forward in time');
        near(row.matrix[0], globeCanvas.width/360); near(row.matrix[3], globeCanvas.height/300);
      }
      const ellipse = globeCtx.ellipses.at(-1), radius = .44*Math.min(globeCanvas.width, globeCanvas.height)*zoom;
      near(ellipse[2], radius*360/globeCanvas.width); near(ellipse[3], radius*300/globeCanvas.height);
      assert.equal(globeCtx.stackDepth, 0); assert.deepEqual(globeCtx.matrix, [1, 0, 0, 1, 0, 0]);
    }
  } finally { globalThis.devicePixelRatio = priorDpr; renderer.destroy(); }
});

check('globe motion stays screen sized across zoom at the same sampled point', () => {
  const data = frame([.0005, 0, 0]); useFrame(data);
  globeCanvas.rect = {left: 20, top: 40, width: 300, height: 300};
  const renderer = globe.create(globeCanvas); app.globeView.renderer = renderer;
  const texture = element('canvas'); texture.width = 2; texture.height = 1; renderer.setTexture(texture);
  let expected;
  try {
    for (const zoom of [1, 2, 4]) {
      app.globeView.pose = {lon: 0, lat: 90, zoom}; renderer.draw(app.globeView.pose);
      const start = globeCtx.strokes.length; app.drawMotion(data, true);
      const center = arrows(globeCtx, start).map(shaft).find(s => Math.abs(s.x-150) < 1e-8 && Math.abs(s.y-150) < 1e-8);
      assert.ok(center); expected ??= center.length; near(center.length, expected);
    }
  } finally { renderer.destroy(); }
});

check('Motion toggle, legend and camera redraw integrate without arrows baked into globe texture', () => {
  resetFlat(); const data = frame([.0003, -.0002, .0001]); useFrame(data);
  const renderer = globe.create(globeCanvas); app.globeView.renderer = renderer;
  const setTexture = renderer.setTexture;
  let textureUploads = 0;
  renderer.setTexture = source => {
    assert.equal(source, mapCanvas);
    assert.equal(arrows(source.getContext('2d')).length, 0, 'No motion strokes in flat texture rendering');
    textureUploads++; setTexture(source);
  };
  mapCtx.strokes = []; app.globeView.active = true; app.globeView.pose = {lon: 10, lat: 90, zoom: 1};
  try {
    get('show-motion').checked = true; get('show-motion').emit('change'); flushRender();
    assert.equal(textureUploads, 1); assert.ok(arrows(globeCtx).length > 0);
    assert.equal(get('motion-legend').hidden, false);
    assert.match(get('motion-legend').innerHTML, /logarithmic/);
    assert.match(get('motion-legend').innerHTML, /30\+/);
    assert.match(get('motion-legend').innerHTML, /cm\/year/);
    const first = globeCtx.strokes.length; app.globeView.pose.lon = 100; app.drawGlobe();
    assert.equal(textureUploads, 1, 'Camera redraw needs no new texture');
    assert.ok(arrows(globeCtx, first).length > 0);
    const last = globeCtx.strokes.length;
    get('show-motion').checked = false; get('show-motion').emit('change'); flushRender();
    assert.equal(get('motion-legend').hidden, true);
    assert.equal(globeCtx.strokes.length, last); assert.equal(globeCtx.visibleStrokes.length, 0);
    app.state.mode = 'initial'; get('show-motion').checked = true; app.updateMotionLegend();
    assert.equal(get('motion-legend').hidden, true);
  } finally { renderer.destroy(); }
});

console.log(`${checks} motion UI checks passed.`);
