/* Exercise the actual form's Lite defaults and saved-world compatibility. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
assert.match(html, /<option value="effective" selected>Lite · effective trench force<\/option>/);
assert.match(html, /<title>Deep Time Lite · Tectonic laboratory<\/title>/);
assert.match(source, /fillConfig\(result\.defaults \|\| result, \{ freshDefaults: true \}\)/);
const nodes = new Map();
for (const match of html.matchAll(/\bid="([^"]+)"/g)) nodes.set(match[1], {
  value:'1', checked:false, options:[{value:'1024'}], add(option) { this.options.push(option); },
  classList:{toggle() {}, add() {}, remove() {}},
});
const get = id => { assert.ok(nodes.has(id), `Control exists: ${id}`); return nodes.get(id); };
function actualFunction(name) {
  const start = source.indexOf(`  function ${name}(`);
  assert.ok(start >= 0);
  return source.slice(start, source.indexOf('\n  function ', start + 1));
}
const sandbox = {$:get, configKeys:['seed','width'], state:{initial:{}}, lipDraft:{events:[]},
  oceanAttachmentDraft:{entries:[],plateCount:null}, isBusy:()=>false, toast() {},
  document:{querySelector:()=>({textContent:''})}, structuredClone,
  normalizeWorldDesign:()=>({version:1,enabled:false}),
  Option:function(text,value) { this.text=text; this.value=value; },
  updateLipDraft() {}, updateRangeOutputs() {},
};
vm.createContext(sandbox);
for (const name of ['readConfig','fillConfig','selectSubductionForceLaw','updateOceanAttachmentNote','updateResolutionNotes','updateRecordedPhysicsNote']) vm.runInContext(actualFunction(name), sandbox);
const json = value => JSON.parse(JSON.stringify(value));
const effective = {seed:12,width:1024,physics_profile:'reviewed_v1',
  effective_subduction:{enabled:true,force_n_per_m:7.25e12},
  subduction_response:'fixed_trench',retained_phases:'disabled',
  primordial_subduction:{enabled:true,initial_slab_depth_km:100,target_margin_fraction:0.35,selection_seed:8},
  primordial_ocean:{enabled:true,plate_count:3},
  continental_lifecycle:{enabled:false},rift_traction:{enabled:false},
  force_limit_rifting:{enabled:true,heal_myr:40},
  trench_persistence:'kinematic',slab_allocation:'uniform'};
sandbox.fillConfig(effective);
let result = sandbox.readConfig();
assert.deepEqual(json(result.effective_subduction), effective.effective_subduction);
assert.equal(get('effective-subduction-force').value, 7.25);
assert.equal(get('effective-subduction-controls').hidden, false);
assert.equal(get('primordial-slab-depth-field').hidden, true);
assert.equal(get('subduction_response').disabled, true);
assert.equal(get('retained_phases').disabled, true);
// Leaving the fresh Lite default selects the normal detailed fresh closures.
get('subduction-force-law').value='slab';
sandbox.selectSubductionForceLaw();
assert.equal(sandbox.readConfig().subduction_response,'moving_hinge_v1');
assert.equal(sandbox.readConfig().retained_phases,'thermal_v1');
// A deliberately chosen detailed draft survives a visit to Lite.
get('subduction_response').value='fixed_trench';
get('retained_phases').value='disabled';
get('subduction-force-law').value='effective';
sandbox.selectSubductionForceLaw();
get('subduction-force-law').value='slab';
sandbox.selectSubductionForceLaw();
assert.equal(sandbox.readConfig().subduction_response,'fixed_trench');
assert.equal(sandbox.readConfig().retained_phases,'disabled');
get('subduction-force-law').value='effective';
sandbox.selectSubductionForceLaw();
assert.match(get('physics-profile-note').textContent, /force from rest/);
assert.match(get('physics-profile-note').textContent, /incoming plate/);
assert.match(get('physics-profile-note').textContent, /Lite · effective trench force/);
assert.doesNotMatch(get('physics-profile-note').textContent, /deep inherited slabs|Moving-trench slab gravity/);
assert.equal(result.primordial_subduction.target_margin_fraction, 0.35);
assert.equal(result.primordial_subduction.selection_seed, 8);
assert.equal(get('effective-force-rifting').checked,true);
assert.deepEqual(json(result.force_limit_rifting),effective.force_limit_rifting);
get('effective-force-rifting').checked=false;
assert.equal(sandbox.readConfig().force_limit_rifting.enabled,false);
assert.equal(sandbox.readConfig().force_limit_rifting.heal_myr,40);
get('subduction-force-law').value='slab';
assert.equal(sandbox.readConfig().force_limit_rifting.enabled,false);
get('subduction-force-law').value='effective';
get('effective-force-rifting').checked=true;
get('effective-subduction-force').value='0';
assert.throws(()=>sandbox.readConfig(), /must be finite and positive/);
get('effective-subduction-force').value='7.25';
// Display-only updates do not discard the law, force or declared geometry.
sandbox.fillConfig({width:1024});
assert.deepEqual(json(sandbox.readConfig().effective_subduction), effective.effective_subduction);
assert.equal(sandbox.readConfig().primordial_subduction.selection_seed, 8);
// Disabled detailed controls cannot silently select incompatible equations.
get('subduction_response').value='moving_hinge_v1';
get('retained_phases').value='thermal_v1';
assert.equal(sandbox.readConfig().subduction_response,'fixed_trench');
assert.equal(sandbox.readConfig().retained_phases,'disabled');
get('primordial-subduction-enabled').checked=false;
assert.throws(()=>sandbox.readConfig(), /needs declared starting subduction margins/);
get('primordial-subduction-enabled').checked=true;
// Choosing this new-run law explicitly replaces inherited detailed preset laws.
// The starting map and declaration selection survive, and switching back restores
// the underlying saved detailed draft.
for (const key of ['continental_lifecycle','rift_traction']) {
  sandbox.state.initial[key]={enabled:true};
  sandbox.fillConfig({[key]:{enabled:true}});
  assert.equal(sandbox.readConfig()[key].enabled,false);
  assert.equal(sandbox.state.initial[key].enabled,true);
  get('subduction-force-law').value='slab';
  assert.equal(sandbox.readConfig()[key].enabled,true);
  get('subduction-force-law').value='effective';
}
for (const [key,value] of [['trench_persistence','attached_slab_v1'],['slab_allocation','fed_v1']]) {
  sandbox.fillConfig({[key]:value});
  assert.equal(sandbox.readConfig()[key],effective[key]);
  get('subduction-force-law').value='slab';
  assert.equal(sandbox.readConfig()[key],value);
  get('subduction-force-law').value='effective';
}
get('subduction-force-law').value='slab';
assert.equal(sandbox.readConfig().subduction_response,'moving_hinge_v1');
assert.equal(sandbox.readConfig().retained_phases,'thermal_v1');
get('subduction-force-law').value='effective';
assert.equal(sandbox.readConfig().primordial_subduction.target_margin_fraction,.35);
assert.equal(sandbox.readConfig().primordial_subduction.selection_seed,8);
assert.equal(get('subduction-response-field').hidden,true);
// Choosing Legacy leaves this form's experiment draft available on return.
get('physics_profile').value='legacy';
sandbox.updateResolutionNotes();
assert.deepEqual(json(sandbox.readConfig().effective_subduction), {enabled:false});
get('physics_profile').value='reviewed_v1';
assert.equal(sandbox.readConfig().effective_subduction.enabled,true);
// Loading a complete older world clears the newer experimental setting.
sandbox.fillConfig({seed:4,width:1024,physics_profile:'reviewed_v1'});
assert.deepEqual(json(sandbox.readConfig().effective_subduction), {enabled:false});
assert.equal(get('effective-subduction-controls').hidden,true);
assert.equal(get('subduction_response').disabled,false);
assert.equal(get('effective-force-rifting').checked,false);
assert.equal(sandbox.state.savedSubductionOptions.trench_persistence,undefined);
// The fresh Highland preset supplies finite selected arcs. Choosing the law
// adopts those declarations until the user explicitly edits their selection.
sandbox.state.initial.initial_subduction={enabled:true,target_margin_fraction:.45,selection_seed:41};
get('subduction-force-law').value='effective';
get('primordial-subduction-enabled').checked=true;
sandbox.updateResolutionNotes();
assert.equal(sandbox.readConfig().primordial_subduction.target_margin_fraction,.45);
assert.equal(sandbox.readConfig().primordial_subduction.selection_seed,41);
sandbox.state.primordialDeclarationEdited=true;
get('primordial-margin-fraction').value='.2';
sandbox.updateResolutionNotes();
assert.equal(sandbox.readConfig().primordial_subduction.target_margin_fraction,.2);
// A real fresh server response enables declarations but does not make their
// fallback selection an explicit user choice. Highland's finite selection wins.
sandbox.state.initial={};
sandbox.fillConfig({seed:42,width:512,physics_profile:'reviewed_v1',
  effective_subduction:{enabled:true,force_n_per_m:3e13},
  primordial_subduction:{enabled:true,target_margin_fraction:1,selection_seed:0},
  subduction_response:'fixed_trench',retained_phases:'disabled',
  continental_lifecycle:{enabled:false},rift_traction:{enabled:false},
  force_limit_rifting:{enabled:false},trench_persistence:'kinematic',slab_allocation:'uniform'},
  {freshDefaults:true});
assert.equal(get('effective-subduction-force').value,30);
assert.equal(sandbox.readConfig().effective_subduction.force_n_per_m,3e13);
assert.equal(sandbox.state.primordialDeclarationEdited,false);
assert.equal(sandbox.readConfig().primordial_subduction.target_margin_fraction,1);
sandbox.state.initial.initial_subduction={enabled:true,target_margin_fraction:.45,selection_seed:41};
sandbox.updateResolutionNotes();
assert.equal(sandbox.readConfig().primordial_subduction.target_margin_fraction,.45);
assert.equal(sandbox.readConfig().primordial_subduction.selection_seed,41);
// Loading a recorded enabled declaration keeps its explicit selection.
sandbox.fillConfig({primordial_subduction:{enabled:true,target_margin_fraction:.2,selection_seed:9}});
sandbox.updateResolutionNotes();
assert.equal(sandbox.state.primordialDeclarationEdited,true);
assert.equal(sandbox.readConfig().primordial_subduction.target_margin_fraction,.2);
assert.equal(sandbox.readConfig().primordial_subduction.selection_seed,9);
// The recorded history explains its own force law, independent of form state.
sandbox.updateRecordedPhysicsNote({width:1024,height:512,physics_profile_version:1,
  effective_subduction_version:1,effective_subduction_diagnostics:{force_n_per_m:5e12}},false);
assert.match(get('recorded-physics-note').textContent,/effective trench force, 5 × 10¹² N\/m, incoming plate only/);
assert.match(get('recorded-physics-note').textContent,/Lite · effective trench force/);
assert.match(get('recorded-physics-note').textContent,/declared initial trench arcs/);
assert.doesNotMatch(get('recorded-physics-note').textContent,/deep slabs|historical fixed-trench response/);
console.log('Effective subduction UI: force units, saved configs, compatible laws and preset overrides passed.');
