/* Unresolved volcanic centers remain source markers, not painted island terrain. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const globe = require('../web/globe.js');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const near = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const xyz = (lon, lat) => {
  const l = lon * Math.PI / 180, a = lat * Math.PI / 180;
  return [Math.cos(a)*Math.cos(l), Math.cos(a)*Math.sin(l), Math.sin(a)];
};
function context() {
  let matrix = [1, 0, 0, 1, 0, 0], points = [];
  const stack = [];
  return {
    strokes: [], visibleStrokes: [], ellipses: [], circles: [], textureReads: [], paints: 0,
    get matrix() { return matrix.slice(); }, get stackDepth() { return stack.length; },
    save() { stack.push({matrix: matrix.slice(), strokeStyle: this.strokeStyle, lineWidth: this.lineWidth}); },
    restore() { assert.ok(stack.length, 'Balanced canvas save/restore'); const old = stack.pop(); matrix = old.matrix; this.strokeStyle = old.strokeStyle; this.lineWidth = old.lineWidth; },
    setTransform(...m) { matrix = m; },
    translate(x, y) { matrix[4] += matrix[0]*x + matrix[2]*y; matrix[5] += matrix[1]*x + matrix[3]*y; },
    scale(x, y) { matrix[0] *= x; matrix[1] *= x; matrix[2] *= y; matrix[3] *= y; },
    beginPath() { points = []; }, moveTo(x, y) { points.push([x, y]); }, lineTo(x, y) { points.push([x, y]); },
    arc(...args) { this.circles.push(args); points.push(args.slice(0, 2)); }, fill() {},
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
  fetch() { throw new Error('Point rendering must not request simulation data'); },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root, 'web/app.js'), 'utf8').replace('  initialize();',
  '  globalThis.app = {state, mapView, globeView, render, drawGlobe, volcanicPoints, updateVolcanicPointNote, projectedVolcanicPoints, drawVolcanicPoints, volcanicPointHover, showHover};'), sandbox);
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
function resetFlat() {
  app.globeView.active = false;
  Object.assign(app.mapView, {zoom: 1, cx: .5, cy: .5});
  mapCanvas.width = 600; mapCanvas.height = 300;
  mapCanvas.rect = {left: 31, top: 47, width: 600, height: 300};
}


const pointFeature = (id, lon = 0, lat = 0, extra = {}) => ({id, owner: 1, owner_uid: 8,
  geometry_xyz: [xyz(lon, lat)], volume_km3: 1250.5, source_time_myr: 170, status: 'pending', ...extra});
function pointFrame(features = [pointFeature(1)]) {
  const data = frame(); data.plates[0].uid = 8; data.plates[0].name = 'Host plate';
  data.arc_point_version = 1; data.volcanic_point_features = features; return data;
}
const amber = (ctx, from = 0) => ctx.strokes.slice(from).filter(row => row.style === '#ffc66d');
const pointer = (lon, lat, data) => ({u: (lon + 180)/360, v: .5-lat/180,
  x: Math.floor((lon+180)/360*data.width), y: Math.floor((.5-lat/180)*data.height)});

check('legacy, empty enabled and editor frames have distinct controls without stale text', () => {
  useFrame(frame()); app.updateVolcanicPointNote(app.state.frame);
  assert.equal(get('show-volcanic-points').disabled, true); assert.equal(get('volcanic-point-note').hidden, true);
  useFrame(pointFrame([])); app.updateVolcanicPointNote(app.state.frame);
  assert.equal(get('show-volcanic-points').disabled, false);
  assert.match(get('volcanic-point-note').textContent, /No unresolved volcanic centers/);
  app.state.mode = 'initial'; app.updateVolcanicPointNote(app.state.frame);
  assert.equal(get('show-volcanic-points').disabled, true); assert.equal(get('volcanic-point-note').textContent, '');
  useFrame(pointFrame()); get('show-volcanic-points').checked = true;
  app.updateVolcanicPointNote(app.state.frame);
  assert.match(get('volcanic-point-note').textContent, /1 volcanic center with pending magma/);
  assert.match(get('volcanic-point-note').textContent, /not island size or height/);
});

check('only supported finite unit-sphere pending points are displayed', () => {
  const data = pointFrame([pointFeature(1), pointFeature(2, 0, 0, {volume_km3: 0}),
    pointFeature(3, 0, 0, {volume_km3: NaN}), pointFeature(4, 0, 0, {geometry_xyz: [[2, 0, 0]]}),
    pointFeature(5, 0, 0, {geometry_xyz: [[1, 0, 0], [0, 1, 0]]}),
    pointFeature(6, 0, 0, {status: 'promoted'}), pointFeature(7, 0, 0, {owner: -1}), null]);
  assert.deepEqual(Array.from(app.volcanicPoints(data), p => p.id), [1]);
  data.arc_point_version = 0; assert.equal(app.volcanicPoints(data).length, 0);
});

check('flat seam markers and precise hover wrap to both visible edges', () => {
  resetFlat(); const data = pointFrame([pointFeature(1, 180, 0)]); useFrame(data);
  get('show-volcanic-points').checked = true;
  const rows = app.projectedVolcanicPoints(data);
  assert.equal(rows.length, 2); near(rows[0].x, 0); near(rows[1].x, 600);
  for (const lon of [-179.9, 179.9]) {
    const hover = app.volcanicPointHover(data, pointer(lon, 0, data)).join(' / ');
    assert.match(hover, /Volcanic center 1/); assert.match(hover, /pending magma: 1,250\.5 km³/);
  }
  assert.equal(app.volcanicPointHover(data, pointer(0, 0, data)).length, 0);
  app.mapView.cx = 1; const centered = app.projectedVolcanicPoints(data);
  assert.equal(centered.length, 1); near(centered[0].x, 300);
});

check('point screen sizes stay constant under map zoom and device pixel ratio', () => {
  resetFlat(); const data = pointFrame(); const before = JSON.stringify(data);
  for (const zoom of [1, 4, 16]) for (const dpr of [1, 2]) {
    app.mapView.zoom = zoom; mapCanvas.width = 600*dpr; mapCanvas.height = 300*dpr;
    const start = mapCtx.strokes.length, circleStart = mapCtx.circles.length; app.drawVolcanicPoints(data);
    const drawn = amber(mapCtx, start); assert.equal(drawn.length, 1);
    near(drawn[0].matrix[0], dpr); near(drawn[0].matrix[3], dpr);
    near(drawn[0].points[0][0], 300); near(drawn[0].points[0][1], 150);
    near(mapCtx.circles[circleStart][2], 3.5); near(mapCtx.circles[circleStart+1][2], .9);
  }
  assert.equal(JSON.stringify(data), before); assert.equal(mapCtx.stackDepth, 0);
  resetFlat();
});

check('polar globe markers reject the back hemisphere and hover the actual projection', () => {
  const data = pointFrame([pointFeature(1, 0, 90), pointFeature(2, 0, -90), pointFeature(3, 90, 80)]);
  useFrame(data); app.globeView.active = true; app.globeView.pose = {lon: 0, lat: 90, zoom: 1};
  globeCanvas.width = 720; globeCanvas.height = 600;
  globeCanvas.rect = {left: 71, top: 19, width: 360, height: 300};
  const rows = app.projectedVolcanicPoints(data, true);
  assert.deepEqual(Array.from(rows, row => row.feature.id), [1, 3]);
  near(rows[0].x, 180); near(rows[0].y, 150);
  assert.ok(rows[1].x > 180); near(rows[1].y, 150);
  const first = globeCtx.strokes.length; app.drawVolcanicPoints(data, true);
  assert.equal(amber(globeCtx, first).length, 2); near(amber(globeCtx, first)[0].matrix[0], 2);
  assert.match(app.volcanicPointHover(data, pointer(0, 90, data)).join(' / '), /Volcanic center 1/);
  assert.equal(app.volcanicPointHover(data, pointer(0, -90, data)).length, 0);
  assert.equal(globeCtx.stackDepth, 0);
  resetFlat();
});

check('hover reports source inventory and owner without inventing height or promotion', () => {
  const data = pointFrame([pointFeature(1, 0, 0, {reason: 'connected_source_origins'})]); useFrame(data);
  const hover = app.volcanicPointHover(data, pointer(0, 0, data)).join(' / ');
  assert.match(hover, /Host: Host plate/); assert.match(hover, /Source supplied: 170 Myr/);
  assert.match(hover, /no island height recorded/); assert.match(hover, /Source lineage: connected source origins/);
  assert.doesNotMatch(hover, /retained|emerged|ready to promote|Awaiting patch/i);
  data.plates[0].id = 5; assert.match(app.volcanicPointHover(data, pointer(0, 0, data)).join(' / '), /Host: Host plate/);
  get('show-volcanic-points').checked = false;
  assert.equal(app.volcanicPointHover(data, pointer(0, 0, data)).length, 0);
  app.updateVolcanicPointNote(data); assert.match(get('volcanic-point-note').textContent, /Turn on Volcanic centers/);
  get('show-volcanic-points').checked = true;
  assert.equal(app.volcanicPointHover(frame(), pointer(0, 0, data)).length, 0);
});

check('a reused plate slot cannot relabel the original pending source host', () => {
  resetFlat(); const data = pointFrame([pointFeature(1, 0, 0, {reason: 'unresolved_advection_host'})]); useFrame(data);
  get('show-volcanic-points').checked = true;
  data.plates = [{id: 1, uid: 99, name: 'Replacement plate'}];
  let hover = app.volcanicPointHover(data, pointer(0, 0, data)).join(' / ');
  assert.match(hover, /Host: Former host UID 8/); assert.doesNotMatch(hover, /Replacement plate/);
  assert.match(hover, /Source lineage: unresolved advection host/);
  data.plates.push({id: 5, uid: 8, name: 'Original plate'});
  hover = app.volcanicPointHover(data, pointer(0, 0, data)).join(' / ');
  assert.match(hover, /Host: Original plate/); assert.doesNotMatch(hover, /Replacement plate/);
  data.volcanic_point_features[0].owner_uid = 0;
  hover = app.volcanicPointHover(data, pointer(0, 0, data)).join(' / ');
  assert.match(hover, /Host: Replacement plate/, 'An unrecorded UID can still use the declared slot');
});

check('markers draw after flat terrain and stay out of the globe texture', () => {
  resetFlat(); const data = pointFrame(); useFrame(data); get('show-volcanic-points').checked = true;
  get('show-motion').checked = false;
  const before = JSON.stringify(data), start = mapCtx.strokes.length;
  app.render(); flushRender(); assert.equal(amber(mapCtx, start).length, 1);
  app.globeView.active = true; app.globeView.pose = {lon: 0, lat: 0, zoom: 1};
  let uploads = 0;
  app.globeView.renderer = {
    setTexture() { uploads++; assert.equal(mapCtx.visibleStrokes.filter(row => row.style === '#ffc66d').length, 0); },
    draw() { globeCtx.visibleStrokes = []; }
  };
  const firstGlobe = globeCtx.strokes.length;
  app.render(); flushRender(); assert.equal(uploads, 1); assert.equal(amber(globeCtx, firstGlobe).length, 1);
  assert.equal(JSON.stringify(data), before);
  get('show-volcanic-points').checked = false; get('show-volcanic-points').emit('change'); flushRender();
  assert.equal(globeCtx.visibleStrokes.filter(row => row.style === '#ffc66d').length, 0);
  assert.equal(get('map-hover').hidden, true);
  get('show-volcanic-points').checked = true;
  useFrame(frame()); app.render(); flushRender();
  assert.equal(globeCtx.visibleStrokes.filter(row => row.style === '#ffc66d').length, 0);
  assert.equal(get('volcanic-point-note').hidden, true);
});
console.log(`${checks} volcanic center UI checks passed`);
