"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("world-map");
  const context = canvas.getContext("2d", { alpha: false });
  const raster = document.createElement("canvas");
  const rasterContext = raster.getContext("2d");
  // Three copies of the raster side by side, padded with one wrapped column at each end.
  // Longitude is periodic, but smoothed drawImage clamps at image edges, so the -180 and
  // +180 columns never blended: the join measured 2.9x a normal column step, a visible seam
  // wherever the map wraps (horizontal pan, yaw-offset views, the globe texture). Drawing
  // separately clipped copies instead left an anti-aliased seam where they met (18-54x);
  // inside one image the join between copies is ordinary interpolation (measured 0-1.7x).
  const rasterWrap = document.createElement("canvas");
  const rasterWrapContext = rasterWrap.getContext("2d");
  const configKeys = ["seed", "width", "plate_count", "duration_myr", "dt_myr", "snapshot_myr", "slab_pull", "ridge_push", "erosion", "rift_strength", "mechanics_nodes", "mesh_level", "coast_geometry_level", "adaptive_refinement", "deforming_regions", "deformation_width_km", "material_face_budget"];
  const boundaryNames = ["None", "Spreading ridge", "Subduction", "Transform", "Continental collision", "Continental rift"];
  const boundaryColors = [null, [119, 212, 220], [239, 128, 108], [197, 161, 222], [244, 211, 138], [237, 155, 226]];
  const crustNames = ["Oceanic crust", "Continental crust", "Stable craton", "Island arc / terrane"];
  const crustColors = [[30, 65, 86], [148, 164, 126], [213, 177, 111], [175, 128, 104]];
  const elevationStops = [-11000, -6500, -4000, -1500, -1, 0, 400, 1200, 2400, 4200, 6500, 9000];
  const elevationColors = [[5, 18, 34], [10, 38, 65], [19, 70, 96], [37, 115, 135], [94, 162, 166], [121, 154, 117], [132, 157, 111], [165, 160, 114], [164, 139, 103], [154, 126, 107], [207, 201, 185], [247, 245, 234]];
  const state = { initial: null, frame: null, frames: [], status: { state: "idle" }, mode: "initial", layer: "crust", brush: 1, brushSize: 8, editorTool: "brush", drawing: false, lastPoint: null, hover: null, measure: { active: false, points: [] }, cache: new Map(), frameRequest: 0, initialRequest: 0, currentIndex: 0, playing: false, playTimer: null, pollBusy: false, pollEpoch: 0, scrubTimer: null, rendering: false, starting: false, initialLoading: false, loadingRun: false, runTransition: false, runId: null, toastTimer: null, hasConnected: false };
  Object.assign(state, { record: null, recordRequest: 0, recordCount: -1, recordFallback: false, inspection: null, inspectionRequest: 0, inspectionPending: false, followLatest: true });
  const mapView = { zoom: 1, cx: .5, cy: .5, drag: null, rasterData: null, rasterLayer: null };
  const globeCanvas = $("world-globe");
  const globeView = { preferred: new URLSearchParams(location.search).get("view") === "globe", active: false,
    renderer: null, pose: { lon: -60, lat: 20, zoom: 1 }, flat: null, drag: null, scheduled: false, textureFrame: null };
  const backarcReview = { runId: null, expanded: new Set(), collapsed: new Set() };
  const trenchReview = { runId: null, selectedId: null };
  const riftReview = { runId: null, selectedId: null };
  const foundationLayers = { thickness: { field: "crustal_thickness_km", control: "layer-thickness" }, basins: { field: "foreland_deflection_m", control: "layer-basins" }, damage: { field: "rift_damage", control: "layer-damage" }, weakness: { field: "rift_strength_relative", control: "layer-weakness" } };
  Object.assign(foundationLayers, { deformation: { field: "deformation_weight", control: "layer-deformation", materialOnly: true },
    strain: { field: "geometric_strain_percent", control: "layer-strain", materialOnly: true },
    refinement: { field: "refinement_level", control: "layer-refinement", materialOnly: true } });
  const terrain = { job: null, result: null, jobs: [], selectionId: null, selectionManual: false, resultRequest: 0, resultLoading: false, listRequest: 0, listLoaded: false, starting: false, cancelling: false, polling: false, epoch: 0, supported: true, viewer: null };
  const terrainImport = { info: null, path: null, loading: false, browsing: false, request: 0 };
  const gospl = { job: null, pending: null, result: null, jobs: [], selectionManual: false, starting: false, cancelling: false, polling: false, listRequest: 0, listLoaded: false, supported: true, rangeRun: null, rangeCount: -1, startIndex: 0, endIndex: 0, endManual: false, notice: "" };
  const terrainCanvas = $("terrain-map");
  const terrainContext = terrainCanvas.getContext("2d", { alpha: false });
  const editHistory = { undo: [], redo: [], before: null, changed: 0, maxBytes: 16 * 1024 * 1024, maxEntries: 8 };
  const timings = { simulation: null, terrain: null, gospl: null };
  const globeOrientation = { runId: undefined, output: { yaw: 0, pitch: 0, roll: 0 }, draftOutput: { yaw: 0, pitch: 0, roll: 0 }, draftInitial: { yaw: 0, pitch: 0, roll: 0 }, pending: false };
  const designEditor = { preview: null, picking: false };
  const lipDraft = { events: [] };
  const oceanAttachmentDraft = { entries: [], plateCount: null };
  Object.assign(foundationLayers, { lip: { field: "lip_deposited_km", control: "layer-lip", materialOnly: true },
    lipheat: { field: "lip_thermal_support_m", control: "layer-lipheat", materialOnly: true } });
  const monotonicNow = () => typeof performance !== "undefined" ? performance.now() : Date.now();

  function durationLabel(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const days = Math.floor(total / 86400), hours = Math.floor(total % 86400 / 3600), minutes = Math.floor(total % 3600 / 60), remainder = total % 60;
    if (days) return `${days}d ${hours}h ${minutes}m`;
    if (hours) return `${hours}h ${minutes}m ${remainder}s`;
    if (minutes) return `${minutes}m ${remainder}s`;
    return `${remainder}s`;
  }

  function captureTiming(channel, status, key) {
    const elapsed = status?.elapsed_seconds;
    if (!key || elapsed == null || !Number.isFinite(Number(elapsed))) { timings[channel] = null; updateTimingDisplays(); return; }
    const phase = status.timing_state || (status.state === "paused" ? "paused" : ["running", "pausing", "resuming"].includes(status.state) ? "estimating" : "complete");
    const previous = timings[channel], received = monotonicNow();
    let seconds = Math.max(0, Number(elapsed));
    if (previous?.key === key && ["running", "estimating"].includes(previous.phase) && ["running", "estimating"].includes(phase)) seconds = Math.max(seconds, previous.elapsed);
    timings[channel] = { key, elapsed: seconds, eta: status.eta_seconds != null && Number.isFinite(Number(status.eta_seconds)) ? Math.max(0, Number(status.eta_seconds)) : null, phase, received, state: status.state };
    updateTimingDisplays();
  }

  function timingLabel(timing, now = monotonicNow()) {
    if (!timing) return "";
    const active = ["running", "estimating"].includes(timing.phase), delta = active ? Math.max(0, now - timing.received) / 1000 : 0;
    const elapsed = `${durationLabel(timing.elapsed + delta)} elapsed`;
    if (timing.phase === "paused") return `${elapsed} · paused`;
    if (!active) return `${elapsed}${timing.state === "complete" ? " · complete" : ""}`;
    if (timing.eta == null) return `${elapsed} · estimating time remaining…`;
    const remaining = Math.max(0, timing.eta - delta);
    return `${elapsed} · ${remaining >= 1 ? `about ${durationLabel(remaining)} remaining` : "updating estimate…"}`;
  }

  function updateTimingDisplays() {
    for (const [channel, id] of [["simulation", "simulation-timing"], ["terrain", "terrain-timing"], ["gospl", "gospl-timing"]]) {
      $(id).textContent = timingLabel(timings[channel]);
      $(id).hidden = !timings[channel];
    }
  }

  function normalizedOrientation(value) {
    const result = {};
    for (const axis of ["yaw", "pitch", "roll"]) { const angle = Number(value?.[axis] ?? 0); result[axis] = Number.isFinite(angle) ? ((angle + 180) % 360 + 360) % 360 - 180 : 0; }
    return result;
  }

  function sameOrientation(a, b) {
    const first = normalizedOrientation(a), second = normalizedOrientation(b);
    return ["yaw", "pitch", "roll"].every((axis) => Math.abs(first[axis] - second[axis]) < 1e-8);
  }

  function syncOrientationContext() {
    if (globeOrientation.runId === state.runId) return;
    globeOrientation.runId = state.runId;
    let saved = null;
    try { if (state.runId) saved = JSON.parse(window.localStorage?.getItem(`deep-time.orientation.${state.runId}`) || "null"); } catch {}
    globeOrientation.output = normalizedOrientation(saved);
    globeOrientation.draftOutput = { ...globeOrientation.output };
    globeOrientation.draftInitial = normalizedOrientation(null);
  }

  function outputOrientation() { syncOrientationContext(); return { ...globeOrientation.output }; }
  function orientationLabel(value) {
    const pose = normalizedOrientation(value);
    return sameOrientation(pose, null) ? "Original globe orientation" : `Yaw ${formatTime(pose.yaw)}° · tilt ${formatTime(pose.pitch)}° · roll ${formatTime(pose.roll)}°`;
  }
  function orientedPath(path, pose = outputOrientation()) {
    return sameOrientation(pose, null) ? path : `${path}${path.includes("?") ? "&" : "?"}orientation=${encodeURIComponent(JSON.stringify(normalizedOrientation(pose)))}`;
  }
  function frameCacheKey(index, pose = outputOrientation(), run = state.runId) {
    const value = normalizedOrientation(pose);
    return `${run || ""}/${index}/${value.yaw},${value.pitch},${value.roll}`;
  }
  function savedOrientationNote(job) {
    const saved = job?.orientation || job?.metadata?.orientation;
    return `${orientationLabel(saved)}${sameOrientation(saved, outputOrientation()) ? "" : " · Orientation differs from current view; build again to match."}`;
  }

  function updateOrientationControls() {
    syncOrientationContext();
    const initial = state.mode === "initial", draft = initial ? globeOrientation.draftInitial : globeOrientation.draftOutput;
    for (const axis of ["yaw", "pitch", "roll"]) {
      $(`orientation-${axis}`).value = draft[axis]; $(`orientation-${axis}-slider`).value = draft[axis];
      $(`orientation-${axis}`).disabled = globeOrientation.pending || state.loadingRun || (initial && isBusy());
      $(`orientation-${axis}-slider`).disabled = $(`orientation-${axis}`).disabled;
    }
    $("orientation-apply").textContent = globeOrientation.pending ? "Rotating globe…" : initial ? "Rotate starting world" : "Apply to view & exports";
    $("orientation-apply").disabled = globeOrientation.pending || state.loadingRun || (initial ? isBusy() || !state.initial || sameOrientation(draft, null) : !state.runId || !state.frames.length || (!!state.frame && sameOrientation(draft, globeOrientation.output)));
    $("orientation-reset").textContent = initial ? "Reset angles" : "Reset orientation";
    $("orientation-reset").disabled = globeOrientation.pending || state.loadingRun || (initial && isBusy()) || (sameOrientation(draft, null) && (initial || sameOrientation(globeOrientation.output, null)));
    $("orientation-description").textContent = initial ? "Rotate the whole sphere before the next simulation. This changes the painted starting map, is undoable, and is included when you export its JSON." : "Reposition the whole globe in this history view and future exports. The recorded experiment stays intact. Apply first to preview the same orientation used by exports.";
    $("orientation-current").textContent = initial ? `Rotation to apply: ${orientationLabel(draft)}` : `Applied: ${orientationLabel(globeOrientation.output)}`;
  }

  async function applyGlobeOrientation() {
    updateOrientationControls();
    if ($("orientation-apply").disabled) return;
    if (state.mode === "initial") {
      const before = copyWorld(state.initial), pose = normalizedOrientation(globeOrientation.draftInitial), request = ++state.initialRequest;
      globeOrientation.pending = true; state.initialLoading = true; setRunningControls(); updateOrientationControls();
      try {
        const world = await api("/api/orient-initial", { initial: before, orientation: pose });
        if (request !== state.initialRequest) return;
        rememberEdit(before); state.initial = copyWorld(world.initial || world); globeOrientation.draftInitial = normalizedOrientation(null);
        state.hover = null; mapView.rasterData = null; mapView.zoom = 1; mapView.cx = .5; mapView.cy = .5;
        toast("Starting world rotated. Undo restores the previous map; exporting its JSON keeps this orientation.");
      } catch (error) { toast(error.message, true); }
      finally { globeOrientation.pending = false; if (request === state.initialRequest) state.initialLoading = false; setRunningControls(); updateOrientationControls(); updateStats(); updateMapZoomControls(); render(); }
      return;
    }
    const run = state.runId, index = state.currentIndex;
    globeOrientation.output = normalizedOrientation(globeOrientation.draftOutput);
    try { window.localStorage?.setItem(`deep-time.orientation.${run}`, JSON.stringify(globeOrientation.output)); } catch {}
    globeOrientation.pending = true; state.cache.clear(); state.frameRequest++; state.frame = null; state.hover = null; $("map-hover").hidden = true; mapView.rasterData = null;
    stopPlayback(); clearInspection(); state.recordRequest++; state.recordCount = -1;
    $("map-empty").textContent = "Rotating the globe on its spherical grid…"; $("map-empty").hidden = false;
    updateOrientationControls(); updateExportButtons(); updateTerrainControls(); reconcileTerrainExperiment(); reconcileGosplResult();
    try { await showFrame(index); await refreshRecord(); }
    finally { globeOrientation.pending = false; if (run === state.runId) { if (!state.frame) $("map-empty").textContent = "Could not load the rotated map. Apply again to retry."; updateOrientationControls(); updateExportButtons(); render(); } }
  }

  async function api(path, data) {
    const response = await fetch(path, data === undefined ? { cache: "no-store" } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data, (_, value) => ArrayBuffer.isView(value) ? Array.from(value) : value) });
    let result;
    try { result = await response.json(); } catch { throw new Error(`The local engine returned an unreadable response (${response.status}).`); }
    if (!response.ok) { const error = new Error(result.error || result.message || `Request failed (${response.status}).`); error.status = response.status; throw error; }
    return result;
  }

  function toast(message, error = false) {
    clearTimeout(state.toastTimer);
    $("toast").textContent = message;
    $("toast").classList.toggle("error", error);
    $("toast").hidden = false;
    state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 10000 : 5000);
  }

  function readConfig() {
    const result = {};
    for (const key of configKeys) result[key] = Number($(key).value);
    result.physics_profile = $("physics_profile").value;
    const reviewed = result.physics_profile === "reviewed_v1";
    const effective = reviewed && $("subduction-force-law").value === "effective";
    result.effective_subduction = effective ? {enabled: true, force_n_per_m: Number($("effective-subduction-force").value) * 1e12} : {enabled: false};
    if (effective && (!Number.isFinite(result.effective_subduction.force_n_per_m) || result.effective_subduction.force_n_per_m <= 0)) throw new Error("Effective trench pull per metre must be finite and positive.");
    result.subduction_response = reviewed && !effective ? $("subduction_response").value : "fixed_trench";
    result.retained_phases = reviewed && !effective ? $("retained_phases").value : "disabled";
    // Preserve the detailed draft; selecting the effective law overrides it
    // only in the new-run request, so switching back restores the saved choices.
    if (reviewed) Object.assign(result, structuredClone(state.savedSubductionOptions || {}));
    if (reviewed && (effective || result.force_limit_rifting !== undefined)) result.force_limit_rifting = {...(result.force_limit_rifting || {}), enabled: $("effective-force-rifting").checked};
    if (effective) {
      result.continental_lifecycle = {...(result.continental_lifecycle || {}), enabled: false};
      result.rift_traction = {...(result.rift_traction || {}), enabled: false};
      result.trench_persistence = "kinematic";
      result.slab_allocation = "uniform";
    }
    // Legacy, or the box unchecked, sends only enabled:false: the historical
    // rupture gate, with the server supplying the unused parameters. Hidden or
    // collapsed parameter inputs then cannot block a start with a stale value.
    result.enhanced_rifting = reviewed && $("enhanced-rifting-enabled").checked ? {
      version: 1,
      enabled: true,
      seed: Number($("enhanced-rifting-seed").value),
      amplitude: Number($("enhanced-rifting-amplitude").value),
      correlation_km: Number($("enhanced-rifting-correlation").value),
      craton_margin_km: Number($("enhanced-rifting-craton-margin").value),
    } : {version: 1, enabled: false};
    result.primordial_subduction = {
      enabled: result.physics_profile === "reviewed_v1" && $("primordial-subduction-enabled").checked,
      initial_slab_depth_km: effective ? 100 : Number($("primordial-slab-depth").value),
      target_margin_fraction: Number($("primordial-margin-fraction").value),
      selection_seed: Number($("primordial-selection-seed").value),
    };
    if (effective && !result.primordial_subduction.enabled) throw new Error("Effective trench force needs declared starting subduction margins. Enable the starting margins below Physics setup.");
    result.primordial_ocean = {
      enabled: result.physics_profile === "reviewed_v1" && $("primordial-ocean-enabled").checked,
      plate_count: Number($("primordial-ocean-count").value),
      half_spreading_rate_cm_yr: Number($("primordial-ocean-age-rate").value),
      maximum_age_myr: Number($("primordial-ocean-max-age").value),
    };
    if (reviewed && oceanAttachmentDraft.entries.length) {
      if (!result.primordial_ocean.enabled) throw new Error("Saved ocean–continent connections require ocean splitting. Enable it again or remove the saved connections below Physics setup.");
      if (result.primordial_ocean.plate_count !== oceanAttachmentDraft.plateCount) throw new Error(`Saved ocean–continent connections identify regions in a ${oceanAttachmentDraft.plateCount}-region ocean. Restore that count or remove the saved connections before changing it.`);
      result.primordial_ocean.continental_attachments = structuredClone(oceanAttachmentDraft.entries);
    }
    result.height = Math.round(result.width / 2);
    result.world_design = normalizeWorldDesign(state.initial?.world_design);
    result.lip_events = { version: 1, enabled: $("lip-enabled").checked, seed: Number($("lip-seed").value),
      generated_count: Number($("lip-generated").value), events: structuredClone(lipDraft.events) };
    return result;
  }

  function fillConfig(config, { freshDefaults = false } = {}) {
    if (!config) return;
    if (config.physics_profile !== undefined || config.seed !== undefined) $("physics_profile").value = config.physics_profile || "legacy";
    if (config.effective_subduction !== undefined || config.seed !== undefined) {
      state.detailedSubductionDraft = null;
      $("subduction-force-law").value = config.effective_subduction?.enabled ? "effective" : "slab";
      $("effective-subduction-force").value = (config.effective_subduction?.force_n_per_m ?? 5e12) / 1e12;
    }
    if (config.force_limit_rifting !== undefined || config.seed !== undefined) $("effective-force-rifting").checked = !!config.force_limit_rifting?.enabled;
    for (const key of ["continental_lifecycle", "rift_traction", "trench_persistence", "slab_allocation", "force_limit_rifting"]) {
      if (config[key] !== undefined || config.seed !== undefined) {
        state.savedSubductionOptions ||= {};
        if (config[key] === undefined) delete state.savedSubductionOptions[key];
        else state.savedSubductionOptions[key] = structuredClone(config[key]);
      }
    }
    if (config.subduction_response !== undefined || config.seed !== undefined) $("subduction_response").value = config.subduction_response ?? "fixed_trench";
    if (config.retained_phases !== undefined || config.seed !== undefined) $("retained_phases").value = config.retained_phases ?? "disabled";
    if (config.enhanced_rifting !== undefined || config.seed !== undefined) {
      // A complete run saved before this control existed ran with enhanced rifting off.
      const rifting = config.enhanced_rifting || {};
      $("enhanced-rifting-enabled").checked = !!rifting.enabled;
      $("enhanced-rifting-seed").value = rifting.seed ?? 37;
      $("enhanced-rifting-amplitude").value = rifting.amplitude ?? 0.25;
      $("enhanced-rifting-correlation").value = rifting.correlation_km ?? 350;
      $("enhanced-rifting-craton-margin").value = rifting.craton_margin_km ?? 150;
    }
    if (config.primordial_subduction !== undefined || config.seed !== undefined) {
      state.primordialDeclarationEdited = !!config.primordial_subduction?.enabled && !freshDefaults;
      $("primordial-subduction-enabled").checked = !!config.primordial_subduction?.enabled;
      $("primordial-slab-depth").value = config.primordial_subduction?.initial_slab_depth_km ?? 100;
      $("primordial-margin-fraction").value = config.primordial_subduction?.target_margin_fraction ?? 1;
      $("primordial-selection-seed").value = config.primordial_subduction?.selection_seed ?? 0;
    }
    if (config.primordial_ocean !== undefined || config.seed !== undefined) {
      $("primordial-ocean-enabled").checked = !!config.primordial_ocean?.enabled;
      $("primordial-ocean-count").value = config.primordial_ocean?.plate_count ?? 4;
      $("primordial-ocean-age-rate").value = config.primordial_ocean?.half_spreading_rate_cm_yr ?? 2;
      $("primordial-ocean-max-age").value = config.primordial_ocean?.maximum_age_myr ?? 180;
      oceanAttachmentDraft.entries = structuredClone(config.primordial_ocean?.continental_attachments || []);
      oceanAttachmentDraft.plateCount = oceanAttachmentDraft.entries.length ? Number($("primordial-ocean-count").value) : null;
    }
    for (const key of configKeys) {
      if (config[key] === undefined || !$(key)) continue;
      if (["width", "mechanics_nodes", "mesh_level", "coast_geometry_level", "adaptive_refinement", "deforming_regions"].includes(key) && !Array.from($(key).options).some((option) => Number(option.value) === Number(config[key]))) {
        const label = key === "width" ? `${config[key]} × ${Math.round(config[key] / 2)} · imported` : `${Number(config[key]).toLocaleString()} · imported`;
        $(key).add(new Option(label, config[key]));
      }
      $(key).value = config[key];
    }
    const lips = config.lip_events || { enabled: false, seed: 37001, generated_count: 0, events: [] };
    $("lip-enabled").checked = !!lips.enabled; $("lip-seed").value = lips.seed ?? 37001;
    $("lip-generated").value = lips.generated_count ?? 0; lipDraft.events = structuredClone(lips.events || []);
    updateLipDraft();
    updateRangeOutputs();
    updateResolutionNotes();
    $("timeline-end").textContent = `${Number($("duration_myr").value).toLocaleString()} Myr`;
  }

  function selectSubductionForceLaw() {
    if ($("subduction-force-law").value === "effective") {
      state.detailedSubductionDraft = {response: $("subduction_response").value, phases: $("retained_phases").value};
      $("primordial-subduction-enabled").checked = true;
    } else {
      const detailed = state.detailedSubductionDraft || {response: "moving_hinge_v1", phases: "thermal_v1"};
      $("subduction_response").value = detailed.response;
      $("retained_phases").value = detailed.phases;
    }
    updateResolutionNotes();
  }

  function updateRangeOutputs() {
    for (const key of ["slab_pull", "ridge_push", "erosion", "rift_strength"]) document.querySelector(`output[for="${key}"]`).textContent = `${Number($(key).value).toFixed(2)}×`;
    state.brushSize = Number($("brush-size").value);
    document.querySelector('output[for="brush-size"]').textContent = `${state.brushSize}° · ${Math.round(state.brushSize * Math.PI / 180 * 6371).toLocaleString()} km`;
    $("snapshot_myr").min = $("dt_myr").value;
  }

  function isRunning() { return state.starting || ["running", "pausing", "resuming"].includes(state.status.state); }
  function isBusy() { return isRunning() || state.runTransition || state.initialLoading || state.loadingRun; }

  function setMode(mode) {
    const previousMode = state.mode;
    state.mode = mode;
    if (mode === "initial" && previousMode !== mode) { mapView.zoom = 1; mapView.cx = .5; mapView.cy = .5; mapView.drag = null; }
    updateMapZoomControls();
    $("map-shell").classList.toggle("editing", mode === "initial" && !isBusy());
    if (mode === "initial") {
      state.frameRequest++;
      if (state.inspectionPending) { state.inspectionRequest++; state.inspectionPending = false; }
      stopPlayback();
      state.layer = "crust";
      $("map-title").textContent = "The world before motion";
      $("mode-badge").textContent = "INITIAL CONDITION";
      $("mode-badge").className = "mode-badge";
      $("map-caption").textContent = "Drag to paint · scroll to zoom · Shift-drag to pan";
      $("editor-hint").textContent = state.status.state === "paused" ? "Edit a new starting world here. The paused experiment keeps its saved checkpoint; Resume uses that checkpoint." : "Paint the map to shape continental crust and stable cratons.";
      $("edit-world").textContent = state.frames.length ? "History ↗" : "Editing";
      $("time-value").textContent = "0";
    } else {
      $("map-title").textContent = "A world in motion";
      $("mode-badge").textContent = state.status.state === "paused" ? "PAUSED EXPERIMENT" : isRunning() ? "SIMULATING" : "RECORDED HISTORY";
      $("mode-badge").className = `mode-badge ${isRunning() ? "running" : "history"}`;
      $("map-caption").textContent = "Click to inspect history · scroll to zoom · Shift-drag to pan";
      $("editor-hint").textContent = "The map shows recorded history. Choose Edit to change the starting world.";
      $("edit-world").textContent = "Edit";
    }
    for (const button of document.querySelectorAll(".brush-button")) button.disabled = mode !== "initial" || isBusy();
    $("brush-size").disabled = mode !== "initial" || isBusy();
    updateEditorControls();
    updateOrientationControls();
    updateFoundationLayers();
    for (const button of document.querySelectorAll("[data-layer]")) {
      button.classList.toggle("active", button.dataset.layer === state.layer);
      button.setAttribute("aria-pressed", String(button.dataset.layer === state.layer));
    }
    $("boundary-legend").style.opacity = mode === "initial" ? ".35" : "1";
    $("show-motion").disabled = mode === "initial";
    updateMotionLegend();
    updateLegend();
    updateExportButtons();
    updateStats();
    updateInspection();
    updateBackArcHistory();
    updateProjectionControls();
    render();
  }

  // A viewing camera is independent of the saved world's/export's orientation.
  function updateProjectionControls() {
    const active = globeView.preferred && state.mode === "history";
    if (active !== globeView.active) {
      if (active) {
        try { globeView.renderer ||= DeepTimeGlobe.create(globeCanvas); }
        catch (error) { globeView.preferred = false; toast(`Globe unavailable: ${error.message}`, true); return updateProjectionControls(); }
        globeView.flat = { zoom: mapView.zoom, cx: mapView.cx, cy: mapView.cy };
        Object.assign(mapView, { zoom: 1, cx: .5, cy: .5, drag: null });
      } else if (globeView.flat && state.mode === "history") Object.assign(mapView, globeView.flat);
      globeView.active = active; globeView.drag = null; state.hover = null;
      $("map-hover").hidden = true;
    }
    $("map-shell").classList.toggle("globe-view", active);
    document.body.classList.toggle("viewing-globe", active);
    globeCanvas.hidden = !active;
    canvas.setAttribute("aria-hidden", String(active));
    $("view-flat").setAttribute("aria-pressed", String(!active));
    $("view-globe").setAttribute("aria-pressed", String(active));
    $("view-globe").disabled = state.mode !== "history";
    $("globe-camera-controls").hidden = !active;
    $("globe-hud").hidden = !active;
    $("globe-note").hidden = !active;
    $("map-caption").textContent = active ? "Drag to rotate · scroll to zoom · click to inspect · arrow keys to rotate" :
      state.mode === "initial" ? "Drag to paint · scroll to zoom · Shift-drag to pan" : "Click to inspect history · scroll to zoom · Shift-drag to pan";
    updateMapZoomControls();
  }

  function drawGlobe() {
    if (!globeView.active || !state.frame || globeView.textureFrame !== state.frame) return;
    globeView.renderer.draw(globeView.pose);
    if ($("show-motion").checked) drawMotion(state.frame, true);
    updateGlobePlayback();
    const {lon, lat} = globeView.pose;
    $("globe-center").textContent = `Centered on ${Math.abs(lat).toFixed(0)}° ${lat < 0 ? "S" : "N"}, ${Math.abs(lon).toFixed(0)}° ${lon < 0 ? "W" : "E"}`;
    globeCanvas.setAttribute("aria-label", `Globe at ${formatTime(state.frame.time_myr)} million years. ${$("globe-center").textContent}. Drag or use arrow keys to rotate; plus and minus zoom; Enter inspects the center; Space plays history.`);
  }

  function updateGlobePlayback() {
    const shown = globeView.active ? globeView.textureFrame : state.frame;
    $("globe-epoch").textContent = shown ? `${formatTime(shown.time_myr)} Myr elapsed` : "Loading history…";
    $("globe-play").textContent = state.playing ? "Ⅱ Pause playback" : "▶ Play history";
    $("globe-play").disabled = state.frames.length < 2;
    $("globe-previous").disabled = !state.frame || state.currentIndex <= 0;
    $("globe-next").disabled = !state.frame || state.currentIndex >= state.frames.length - 1;
  }

  function moveGlobe(lon, lat, zoom = globeView.pose.zoom) {
    $("map-hover").hidden = true;
    globeView.pose = {lon: ((lon + 180) % 360 + 360) % 360 - 180, lat: Math.max(-90, Math.min(90, lat)), zoom: Math.max(1, Math.min(4, zoom))};
    updateMapZoomControls();
    if (globeView.scheduled) return;
    globeView.scheduled = true;
    requestAnimationFrame(() => { globeView.scheduled = false; drawGlobe(); });
  }

  function globePoint(event) {
    if (!globeView.active || globeView.scheduled || !state.frame || globeView.textureFrame !== state.frame || state.frames[state.currentIndex]?.time_myr !== state.frame.time_myr) return null;
    const point = globeView.renderer.pick(event.clientX, event.clientY, globeView.pose);
    // The picker already resolves an exact u,v on the sphere; keep it beside the
    // floored cell so the distance measure reads true position, not cell centre.
    return point ? { x: Math.min(state.frame.width - 1, Math.floor(point.u * state.frame.width)),
      y: Math.min(state.frame.height - 1, Math.floor(point.v * state.frame.height)),
      u: point.u, v: point.v } : null;
  }

  function terrainColor(value) {
    if (value <= elevationStops[0]) return elevationColors[0];
    for (let j = 1; j < elevationStops.length; j++) {
      if (value <= elevationStops[j]) {
        const t = (value - elevationStops[j - 1]) / (elevationStops[j] - elevationStops[j - 1]);
        const a = elevationColors[j - 1], b = elevationColors[j];
        return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
      }
    }
    return elevationColors[elevationColors.length - 1];
  }

  const platePalette = new Map();
  function plateColor(id) {
    if (platePalette.has(id)) return platePalette.get(id);
    const hue = ((id * 137.508 + 18) % 360) / 60;
    const c = .39, x = c * (1 - Math.abs(hue % 2 - 1)), m = .29;
    const triplets = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]];
    const value = triplets[Math.floor(hue)].map((v) => (v + m) * 255);
    platePalette.set(id, value);
    return value;
  }

  const domainMetadata = new WeakMap();
  function hasDomains(data) {
    return Boolean(data?.domain && data.domain.length === data.width * data.height);
  }
  function domainAt(data, cell) {
    if (!hasDomains(data) || !Number.isInteger(data.domain[cell]) || data.domain[cell] < 0) return null;
    let entries = domainMetadata.get(data);
    if (!entries) { entries = new Map((data.domains || []).map((domain) => [domain.uid, domain])); domainMetadata.set(data, entries); }
    const uid = data.domain[cell];
    return entries.get(uid) || { uid, name: `Domain ${uid}`, plate_id: data.plate[cell] };
  }
  function plateAt(data, cell) {
    return (data.plates || []).find((plate) => plate.id === data.plate[cell]);
  }
  function regionName(point) {
    return point.domain_name || (Number.isInteger(point.domain_uid) && point.domain_uid >= 0 ? `Domain ${point.domain_uid}` : point.plate_name || `Plate ${point.plate_id}`);
  }
  function regionIdentity(point) {
    return `${point.domain_uid ?? "legacy"}/${point.plate_uid ?? point.plate_id}`;
  }

  function riftLoadingInfo(plate) {
    const diagnostic = plate?.internal_loading;
    if (!diagnostic) return { kind: "unavailable", ratio: null, diagnostic: null };
    if (diagnostic.model === "local material extension and damage") return { kind: "local", ratio: null, diagnostic };
    const load = diagnostic.accumulated_load_myr, threshold = diagnostic.rupture_threshold_myr;
    const ratio = Number.isFinite(load) && Number.isFinite(threshold) && threshold > 0 ? Math.max(0, load / threshold) : null;
    const kind = diagnostic.initial_ocean_protected ? "protected" : diagnostic.eligible === false ? "ineligible" : diagnostic.eligible === true && ratio !== null ? "loading" : "unavailable";
    return { kind, ratio, diagnostic };
  }

  function riftLoadingColor(info) {
    if (info.kind === "local") return [100, 124, 113];
    if (info.kind === "protected") return [128, 129, 116];
    if (info.kind === "ineligible") return [45, 56, 65];
    if (info.kind === "unavailable") return [89, 99, 108];
    const t = Math.min(1, Math.max(0, info.ratio));
    const a = t < .5 ? [49, 98, 115] : [204, 165, 92], b = t < .5 ? [204, 165, 92] : [224, 101, 74];
    const blend = t < .5 ? t * 2 : (t - .5) * 2;
    return a.map((value, i) => Math.round(value + (b[i] - value) * blend));
  }

  function riftLoadingHover(plate) {
    const info = riftLoadingInfo(plate), diagnostic = info.diagnostic;
    if (info.kind === "unavailable") return ["Rift loading was not recorded at this epoch."];
    if (info.kind === "local") {
      const values = ["Continental breakup follows local stretching and material damage, without a plate-wide rupture clock.",
        "Select Rift damage or Lithosphere strength and review the continental rift systems for local evidence."];
      if (typeof diagnostic.eligible === "boolean") values.push(diagnostic.eligible ? "Plate is currently eligible for continental separation." : "Plate is not currently eligible for continental separation.");
      return values;
    }
    const percent = info.ratio === null ? "Load unavailable" : `Accumulated rift load ${Math.round(info.ratio * 100)}% of threshold`;
    let condition = diagnostic.initial_ocean_protected ? "initial ocean protected from internal breakup" : diagnostic.eligible === false ? "currently ineligible (plate size / continental area)" : "eligible for internal breakup";
    if (!diagnostic.initial_ocean_protected && Number.isFinite(diagnostic.cooldown_remaining_myr) && diagnostic.cooldown_remaining_myr > 0) condition += ` · ${formatTime(diagnostic.cooldown_remaining_myr)} Myr cooldown`;
    const values = [`${percent} · ${condition}`];
    const speed = (value) => Number.isFinite(value) ? value.toFixed(1) : "—";
    values.push(`Differing boundary motion ${speed(diagnostic.incompatible_speed_km_myr)} km/Myr · extension ${speed(diagnostic.tensile_speed_km_myr)} · compression ${speed(diagnostic.compressive_speed_km_myr)} km/Myr`);
    const surrounded = Number.isFinite(diagnostic.surrounded_neighbors) && diagnostic.surrounded_neighbors > 0 ? ` · ${diagnostic.surrounded_neighbors} surrounded neighbour${diagnostic.surrounded_neighbors === 1 ? "" : "s"}` : "";
    if (Number.isFinite(diagnostic.load_proxy)) values.push(`Current loading indicator ${diagnostic.load_proxy.toFixed(2)}${surrounded} · relative breakup indicator, not physical stress`);
    return values;
  }

  // Report the actual saved grid behind both the map and the globe texture.
  window.deepTimeDisplayState = () => {
    const data = state.mode === "initial" ? state.initial : state.frame;
    const src = mapView.rasterSource;
    return { width: src?.width ?? null, height: src?.height ?? null, layer: state.layer,
             recordedWidth: data?.width ?? null, recordedHeight: data?.height ?? null,
             source: src ? (state.mode === "initial" ? "initial-grid" : "recorded-frame") : null,
             projection: globeView.active ? "globe" : "map", upgradable: false, pending: false,
             frameIndex: src?.index ?? null };
  };

  function render() {
    if (state.rendering) return;
    state.rendering = true;
    requestAnimationFrame(() => {
      state.rendering = false;
      const data = state.mode === "initial" ? state.initial : state.frame;
      updateSlabWindowNote(data);
      if (!data) return;
      const w = data.width, h = data.height;
      const initial = state.mode === "initial";
      updateRecordedPhysicsNote(data, initial);
      // Saved rasters sample the moving material surface. Keep the picture, inspection,
      // statistics and every layer on that same grid; control-face samples describe a
      // different, coarser surface and must not replace it during rendering.
      const src = data;
      const rw = src.width, rh = src.height, n = rw * rh;
      const showBoundaries = !initial && $("show-boundaries").checked;
      updateSubductionNote(data, showBoundaries);
      if (mapView.rasterSource !== src || mapView.rasterLayer !== state.layer) {
      raster.width = rw; raster.height = rh;
      const picture = rasterContext.createImageData(rw, rh);
      const pixels = picture.data;
      const loadingColors = state.layer === "loading" ? new Map((data.plates || []).map((plate) => [plate.id, riftLoadingColor(riftLoadingInfo(plate))])) : null;
      const unavailableLoadingColor = state.layer === "loading" ? riftLoadingColor({ kind: "unavailable" }) : null;
      for (let i = 0; i < n; i++) {
        const crust = src.crust ? src.crust[i] : 0;
        let color;
        if (initial || state.layer === "crust") color = crustColors[crust] || crustColors[0];
        else if (state.layer === "plate") color = plateColor(hasDomains(src) && src.domain[i] >= 0 ? src.domain[i] : src.plate[i]);
        else if (state.layer === "loading") color = loadingColors.get(src.plate[i]) || unavailableLoadingColor;
        else if (foundationLayers[state.layer]) color = foundationColor(state.layer, foundationLayers[state.layer].materialOnly && crust === 0 ? undefined : data[foundationLayers[state.layer].field]?.[i]);
        else if (state.layer === "age") {
          if (crust === 1 || crust === 2) color = [99, 110, 99];
          else {
            const t = Math.min(1, Math.max(0, src.age[i] / 200));
            color = t < .5 ? [227 - 228 * t, 172 + 10 * t, 104 + 136 * t] : [113 - 144 * (t - .5), 177 - 176 * (t - .5), 172 - 86 * (t - .5)];
          }
        } else color = terrainColor(src.elevation[i]);
        let shade = 1;
        if (!initial && state.layer === "elevation") {
          const x = i % rw, y = Math.floor(i / rw);
          const east = y * rw + (x + 1) % rw, west = y * rw + (x - 1 + rw) % rw;
          const north = Math.max(0, y - 1) * rw + x, south = Math.min(rh - 1, y + 1) * rw + x;
          shade = Math.max(.72, Math.min(1.18, 1 - (src.elevation[east] - src.elevation[west]) * .0000275 - (src.elevation[south] - src.elevation[north]) * .0000375));
        }
        const x = i % rw;
        const right = Math.floor(i / rw) * rw + (x + 1) % rw;
        const below = i + rw < n ? i + rw : i;
        const isLand = initial ? crust !== 0 : src.elevation[i] >= 0;
        const neighborLand = initial ? src.crust[right] !== 0 : src.elevation[right] >= 0;
        const belowLand = initial ? src.crust[below] !== 0 : src.elevation[below] >= 0;
        if (state.layer !== "loading" && !foundationLayers[state.layer] && isLand && (!neighborLand || !belowLand)) shade *= .76;
        let r = color[0] * shade, g = color[1] * shade, b = color[2] * shade;
        pixels[4 * i] = r; pixels[4 * i + 1] = g; pixels[4 * i + 2] = b; pixels[4 * i + 3] = 255;
      }
      rasterContext.putImageData(picture, 0, 0);
      rasterWrap.width = 3 * rw + 2; rasterWrap.height = rh;
      for (let k = 0; k < 3; k++) rasterWrapContext.drawImage(raster, 1 + k * rw, 0);
      rasterWrapContext.drawImage(raster, rw - 1, 0, 1, rh, 0, 0, 1, rh);
      rasterWrapContext.drawImage(raster, 0, 0, 1, rh, 3 * rw + 1, 0, 1, rh);
      mapView.rasterData = data; mapView.rasterSource = src; mapView.rasterLayer = state.layer;
      }
      const displayWidth = globeView.active ? 2048 : Math.max(640, Math.round(canvas.getBoundingClientRect().width * Math.min(window.devicePixelRatio || 1, 2)));
      if (canvas.width !== displayWidth || canvas.height !== Math.round(displayWidth * h / w)) {
        canvas.width = displayWidth; canvas.height = Math.round(displayWidth * h / w);
      }
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.fillStyle = "#0c1821"; context.fillRect(0, 0, canvas.width, canvas.height);
      // The base map is ONE image: the three-copy strip, positioned so its middle copy spans
      // [0, width]. The globe texture takes just the middle copy with its wrapped neighbour
      // columns, so the texture's own edges blend across the antimeridian too (the globe
      // handoff resets the view to zoom 1, cx .5).
      const cell = canvas.width / rw;
      context.save();
      context.translate(canvas.width * (.5 - mapView.cx * mapView.zoom), canvas.height * (.5 - mapView.cy * mapView.zoom)); context.scale(mapView.zoom, mapView.zoom);
      context.imageSmoothingEnabled = state.layer === "elevation" && !initial;
      if (globeView.active) context.drawImage(rasterWrap, rw, 0, rw + 2, rh, -cell, 0, canvas.width + 2 * cell, canvas.height);
      else context.drawImage(rasterWrap, 0, 0, 3 * rw + 2, rh, -canvas.width - cell, 0, 3 * canvas.width + 2 * cell, canvas.height);
      context.restore();
      // Overlays are vector marks over an opaque base, so they simply repeat for each copy
      // that intersects the view: at most two when panned across the antimeridian.
      const halfView = .5 / mapView.zoom;
      const copies = globeView.active ? [0]
        : [-1, 0, 1].filter(k => k + 1 > mapView.cx - halfView && k < mapView.cx + halfView);
      for (const copy of copies) {
      context.save();
      context.translate(canvas.width * (.5 - mapView.cx * mapView.zoom), canvas.height * (.5 - mapView.cy * mapView.zoom)); context.scale(mapView.zoom, mapView.zoom);
      context.translate(copy * canvas.width, 0);
      context.imageSmoothingEnabled = state.layer === "elevation" && !initial;
      drawWorldDesign(data, initial);
      if (showBoundaries && data.boundary && data.plate) {
        const nativeBoundary = Array.isArray(data.boundary_segments);
        const subduction = nativeBoundary ? null : subductionGeometry(data);
        if (mapView.boundaryData !== data || mapView.boundaryWidth !== canvas.width) {
        const paths = Array.from({ length: 6 }, () => new Path2D());
        const sx = canvas.width / w, sy = canvas.height / h;
        if (nativeBoundary) {
          for (const line of DeepTimeNativeBoundaries.paths(data)) {
            line.points.forEach(([x,y], i) => {
              if (i === 0) paths[line.code].moveTo(x*canvas.width,y*canvas.height);
              else paths[line.code].lineTo(x*canvas.width,y*canvas.height);
            });
          }
        } else {
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const i = y * w + x, east = y * w + (x + 1) % w;
            if (data.plate[i] !== data.plate[east]) {
              const code = subduction.keys.has(`vertical:${x}:${y}`) ? 2 : data.boundary[i] || data.boundary[east];
              if (paths[code] && code) {
                paths[code].moveTo((x + 1) * sx, y * sy); paths[code].lineTo((x + 1) * sx, (y + 1) * sy);
                if (x === w - 1) { paths[code].moveTo(0, y * sy); paths[code].lineTo(0, (y + 1) * sy); }
              }
            }
            if (y < h - 1 && data.plate[i] !== data.plate[i + w]) {
              const code = subduction.keys.has(`horizontal:${x}:${y}`) ? 2 : data.boundary[i] || data.boundary[i + w];
              if (paths[code] && code) { paths[code].moveTo(x * sx, (y + 1) * sy); paths[code].lineTo((x + 1) * sx, (y + 1) * sy); }
            }
          }
        }
        }
        mapView.boundaryData = data; mapView.boundaryWidth = canvas.width; mapView.boundaryPaths = paths;
        }
        context.save();
        context.lineWidth = Math.max(1, 1.1 * canvas.width / canvas.getBoundingClientRect().width) / mapView.zoom;
        context.lineJoin = "round"; context.globalAlpha = .9;
        for (const code of [3, 1, 5, 2, 4]) { context.strokeStyle = `rgb(${boundaryColors[code].join(",")})`; context.stroke(mapView.boundaryPaths[code]); }
        context.restore();
        drawSubductionTeeth(data);
      }
      if ($("show-grid").checked) {
        context.save(); context.strokeStyle = "rgba(190,218,229,.17)"; context.lineWidth = Math.max(1, canvas.width / 1200); context.setLineDash([3, 5]);
        for (let j = 1; j < 12; j++) { context.beginPath(); context.moveTo(j * canvas.width / 12, 0); context.lineTo(j * canvas.width / 12, canvas.height); context.stroke(); }
        for (let j = 1; j < 6; j++) { context.beginPath(); context.moveTo(0, j * canvas.height / 6); context.lineTo(canvas.width, j * canvas.height / 6); context.stroke(); }
        context.restore();
      }
      if (!initial && $("show-slab-windows").checked) drawSlabWindows(data);
      if (!initial && $("show-lip").checked) drawLipSources(data);
      if (!initial && state.inspection) drawTrace();
      if (initial && state.hover && !isBusy()) {
        context.save(); context.fillStyle = "rgba(250,235,183,.24)"; context.strokeStyle = "rgba(245,237,214,.95)"; context.lineWidth = 1.2 * canvas.width / 1000;
        if (state.editorTool === "brush") {
          for (const row of sphericalBrushRows(data, state.hover, state.brushSize)) {
            for (const [start, stop] of row.spans) context.fillRect(start / w * canvas.width, row.y / h * canvas.height, (stop - start) / w * canvas.width, canvas.height / h);
          }
        }
        const centerX = (state.hover.x + .5) / w * canvas.width, centerY = (state.hover.y + .5) / h * canvas.height;
        context.beginPath(); context.moveTo(centerX - 5, centerY); context.lineTo(centerX + 5, centerY); context.moveTo(centerX, centerY - 5); context.lineTo(centerX, centerY + 5); context.stroke();
        context.restore();
      }
      context.restore();
      }
      if (!initial && !globeView.active && $("show-motion").checked) drawMotion(data);
      if (globeView.active) { globeView.renderer.setTexture(canvas); globeView.textureFrame = data; drawGlobe(); }
      $("map-empty").hidden = true;
    });
  }

  function updateSubductionNote(data, visible) {
    const note = $("subduction-note");
    note.hidden = !visible;
    if (!visible) return;
    if (Array.isArray(data.boundary_segments)) {
      note.textContent = "Coral triangles mark subduction and point toward the overriding plate. Cyan lines mark spreading ridges.";
      return;
    }
    note.textContent = Array.isArray(data.trench_systems) ?
      "Subduction teeth sit on the overriding plate and point in the direction the slab sinks. Placement follows the recorded trench polarity at this grid resolution." :
      "This older history did not record subduction polarity, so subduction boundaries are shown without teeth.";
  }

  function subductionGeometry(data) {
    if (mapView.subductionData !== data) {
      mapView.subductionGeometry = DeepTimeSubduction.build(data);
      mapView.subductionGeometry.keys = new Set(mapView.subductionGeometry.faces.map(face => face.key));
      mapView.subductionData = data;
    }
    return mapView.subductionGeometry;
  }

  function drawSubductionTeeth(data) {
    const ratio = canvas.width / canvas.getBoundingClientRect().width;
    const options = {
      width: canvas.width, height: canvas.height, zoom: mapView.zoom,
      cx: mapView.cx, cy: mapView.cy, pixelRatio: ratio
    };
    const teeth = Array.isArray(data.boundary_segments) ? DeepTimeNativeBoundaries.triangles(data, options)
      : DeepTimeSubduction.triangles(subductionGeometry(data).faces, options);
    if (!teeth.length) return;
    context.save(); context.beginPath(); context.rect(0, 0, canvas.width, canvas.height); context.clip();
    context.fillStyle = `rgb(${boundaryColors[2].join(",")})`;
    context.strokeStyle = "rgba(12,24,33,.95)"; context.lineWidth = 1 * ratio / mapView.zoom;
    context.beginPath();
    for (const [a, b, tip] of teeth) {
      context.moveTo(...a); context.lineTo(...b); context.lineTo(...tip); context.closePath();
    }
    context.fill(); context.stroke(); context.restore();
  }

  function activeRidgeEpisodes(data) {
    const time = Number(data?.time_myr);
    if (!Number.isFinite(time) || !Array.isArray(data?.ridge_episodes)) return [];
    return data.ridge_episodes.filter((episode) => {
      const center = episode?.center;
      return Array.isArray(center) && center.length === 3 && center.every(Number.isFinite) && Math.hypot(...center) > 1e-12 &&
        Number.isFinite(episode.radius_km) && episode.radius_km > 0 && episode.radius_km < Math.PI * 6371 &&
        Number.isFinite(episode.started_myr) && Number.isFinite(episode.end_myr) && time >= episode.started_myr && time < episode.end_myr;
    });
  }

  function slabWindowProjection(episode, samples = 128) {
    // Sample a true small circle on the sphere. Longitude is then unwrapped,
    // so seam crossings and circles enclosing a pole never draw a map-wide chord.
    const length = Math.hypot(...episode.center), center = episode.center.map((v) => v / length);
    const axis = Math.abs(center[2]) < .9 ? [0, 0, 1] : [1, 0, 0];
    const dot = center.reduce((sum, v, i) => sum + v * axis[i], 0);
    const tangent = axis.map((v, i) => v - dot * center[i]), norm = Math.hypot(...tangent);
    const u = tangent.map((v) => v / norm), v = [center[1] * u[2] - center[2] * u[1], center[2] * u[0] - center[0] * u[2], center[0] * u[1] - center[1] * u[0]];
    const angle = episode.radius_km / 6371, c = Math.cos(angle), s = Math.sin(angle);
    const project = (xyz) => [(Math.atan2(xyz[1], xyz[0]) + Math.PI) / (2 * Math.PI), .5 - Math.asin(Math.max(-1, Math.min(1, xyz[2]))) / Math.PI];
    const ring = [];
    for (let i = 0; i <= samples; i++) {
      const t = i * 2 * Math.PI / samples;
      const point = project(center.map((value, j) => c * value + s * (Math.cos(t) * u[j] + Math.sin(t) * v[j])));
      if (ring.length) { while (point[0] - ring.at(-1)[0] > .5) point[0] -= 1; while (point[0] - ring.at(-1)[0] < -.5) point[0] += 1; }
      ring.push(point);
    }
    return { center: project(center), ring };
  }

  function updateSlabWindowNote(data) {
    const recorded = state.mode !== "initial" && Array.isArray(data?.ridge_episodes), count = activeRidgeEpisodes(data).length;
    const control = $("show-slab-windows"), note = $("slab-window-note");
    control.disabled = !recorded;
    note.hidden = !recorded || !count;
    note.textContent = count ? `${count} active slab window${count === 1 ? "" : "s"} · ${control.checked ? "Dotted rings show approximate thermal footprints." : "Turn on Slab windows to show their footprints."} Visit a ridge-subduction event to review its onset.` : "";
  }

  function drawSlabWindows(data) {
    const episodes = activeRidgeEpisodes(data);
    if (!episodes.length) return;
    const scale = canvas.width / canvas.getBoundingClientRect().width / mapView.zoom;
    context.save(); context.beginPath(); context.rect(0, 0, canvas.width, canvas.height); context.clip();
    context.strokeStyle = "#ffd39a"; context.fillStyle = "#ffd39a"; context.lineWidth = 1.4 * scale;
    context.shadowColor = "#15232c"; context.shadowBlur = 3 * scale;
    context.font = `600 ${10 * scale}px "Segoe UI", sans-serif`; context.textBaseline = "bottom";
    for (const episode of episodes) {
      const geometry = slabWindowProjection(episode);
      context.globalAlpha = Number.isFinite(episode.heat) ? .45 + .5 * Math.max(0, Math.min(1, episode.heat)) : .8;
      context.setLineDash([2 * scale, 4 * scale]);
      for (const wrap of [-1, 0, 1]) {
        context.beginPath();
        geometry.ring.forEach(([x, y], i) => { if (i) context.lineTo((x + wrap) * canvas.width, y * canvas.height); else context.moveTo((x + wrap) * canvas.width, y * canvas.height); });
        context.stroke();
      }
      context.setLineDash([]);
      for (const wrap of [-1, 0, 1]) {
        const x = (geometry.center[0] + wrap) * canvas.width, y = geometry.center[1] * canvas.height;
        context.beginPath(); context.arc(x, y, 2.5 * scale, 0, 2 * Math.PI); context.fill();
        const label = "Slab window", labelWidth = context.measureText(label).width;
        const right = x + 7 * scale + labelWidth > canvas.width;
        context.textAlign = right ? "right" : "left";
        const labelY = y < 18 * scale ? y + 18 * scale : y - 6 * scale;
        context.fillText(label, x + (right ? -7 : 7) * scale, labelY);
      }
    }
    context.restore();
  }

  function updateLipDraft() {
    const selected = $("lip-authored"), previous = selected.value;
    selected.replaceChildren();
    for (const row of lipDraft.events) selected.add(new Option(`${row.id} · ${row.start_myr} Myr · ${row.volume_km3.toLocaleString()} km³`, row.id));
    if (!lipDraft.events.length) selected.add(new Option("No authored provinces", ""));
    else if (lipDraft.events.some(row => row.id === previous)) selected.value = previous;
    $("lip-remove").disabled = !lipDraft.events.length;
    $("lip-draft-note").textContent = `${lipDraft.events.length} authored province${lipDraft.events.length === 1 ? "" : "s"}. Changes apply to a new experiment.`;
  }

  function addLipEvent() {
    for (const input of document.querySelectorAll(".lip-controls input")) if (!input.reportValidity()) return;
    const id = $("lip-name").value.trim();
    if (!id || !/^[A-Za-z0-9_-]{1,64}$/.test(id)) { toast("Choose a province name using letters, numbers, hyphens or underscores."); return; }
    if (lipDraft.events.some(row => row.id === id)) { toast("That province name is already in this scenario."); return; }
    if (lipDraft.events.length + Number($("lip-generated").value) >= 16) { toast("A scenario supports up to 16 authored and generated LIPs together."); return; }
    const row = { id };
    for (const [key, suffix] of Object.entries({ lon_deg: "lon", lat_deg: "lat", radius_km: "radius", start_myr: "start", duration_myr: "duration", volume_km3: "volume", intrusive_fraction: "intrusive", thermal_support_m: "heat", cooling_myr: "cooling" })) row[key] = Number($("lip-" + suffix).value);
    lipDraft.events.push(row); $("lip-enabled").checked = true; updateLipDraft();
    $("lip-name").value = `province-${lipDraft.events.length + 1}`;
  }

  function updateLipHistory() {
    const data = state.mode === "history" ? state.frame : null;
    const events = data?.lip_version === 1 && Array.isArray(data.lip_events) ? data.lip_events : null;
    $("lip-history").hidden = !events; $("show-lip").disabled = !events;
    const rows = $("lip-history-rows"); rows.replaceChildren();
    if (!events) return;
    if (!events.length) { rows.textContent = "This experiment has LIPs enabled with no scheduled sources."; return; }
    for (const row of events) {
      const phase = data.time_myr < row.start_myr ? "scheduled" : data.time_myr < row.start_myr + row.duration_myr ? "active source" : "source ended";
      const item = document.createElement("p");
      const amount = value => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 0 });
      item.textContent = `${row.id} · ${phase} · ${row.start_myr.toFixed(2)}–${(row.start_myr + row.duration_myr).toFixed(2)} Myr. Supplied ${amount(row.supplied_volume_km3)} km³; intrusive ${amount(row.intrusive_volume_km3)}, extrusive ${amount(row.extrusive_volume_km3)}, pending ${amount(row.pending_volume_km3)}, rejected ${amount(row.rejected_volume_km3)} km³. Unoccupied or unsupported footprint and full columns retain their rejected share.`;
      rows.append(item);
    }
  }

  function drawLipSources(data) {
    const events = (data?.lip_events || []).filter(row => data.time_myr >= row.start_myr && data.time_myr < row.start_myr + row.duration_myr);
    if (!events.length) return;
    const scale = canvas.width / canvas.getBoundingClientRect().width / mapView.zoom;
    context.save(); context.strokeStyle = "#fa8bdc"; context.fillStyle = "#fa8bdc";
    context.lineWidth = 1.5 * scale; context.setLineDash([5 * scale, 3 * scale]);
    context.font = `600 ${10 * scale}px "Segoe UI", sans-serif`;
    for (const row of events) {
      const lon = row.lon_deg * Math.PI / 180, lat = row.lat_deg * Math.PI / 180;
      const center = [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
      const geometry = slabWindowProjection({ center, radius_km: row.radius_km }, 48);
      for (const wrap of [-1, 0, 1]) {
        context.beginPath(); geometry.ring.forEach(([x, y], i) => { if (i) context.lineTo((x + wrap) * canvas.width, y * canvas.height); else context.moveTo((x + wrap) * canvas.width, y * canvas.height); }); context.stroke();
        context.fillText(`LIP · ${row.id}`, (geometry.center[0] + wrap) * canvas.width + 4 * scale, geometry.center[1] * canvas.height - 4 * scale);
      }
    }
    context.restore();
  }

  function updateLegend() {
    const legend = $("layer-legend");
    const loading = state.mode === "history" && state.layer === "loading";
    legend.classList.toggle("loading-legend", loading);
    $("loading-map-note").hidden = !loading;
    const localMechanics = (state.frame?.plates || []).some((plate) => plate.internal_loading?.model === "local material extension and damage");
    $("loading-map-note").textContent = localMechanics ? "Continental breakup follows local stretching and damage; select Rift damage to see its distribution. The plate-wide loading scale below applies only to oceanic breakup. Hover to distinguish the recorded model for each plate."
      : "Differing boundary motions can load a plate even when its net movement is small. Colors show accumulated loading as a relative breakup indicator, not physical stress. Hover for loading and cooldown details.";
    if (state.mode === "initial" || state.layer === "crust") {
      legend.innerHTML = '<div class="legend-swatches"><span><i style="background:#1e4156"></i>Ocean</span><span><i style="background:#94a47e"></i>Continent</span><span><i style="background:#d5b16f"></i>Craton</span>' + (state.mode === "history" ? '<span><i style="background:#af8068"></i>Arc / terrane</span>' : '') + '</div>';
    } else if (loading) {
      legend.innerHTML = '<div class="gradient" style="background:linear-gradient(90deg,#316273,#cca55c,#e0654a)"></div><div class="legend-labels"><span>0% LOADED</span><span>50%</span><span>100%+ OF THRESHOLD</span></div><div class="legend-swatches loading-status-legend"><span><i style="background:#59636c"></i>Not recorded</span><span><i style="background:#2d3841"></i>Ineligible</span><span><i style="background:#808174"></i>Protected ocean</span></div>';
      if (localMechanics) legend.innerHTML += '<div class="legend-swatches"><span><i style="background:#647c71"></i>Local continental mechanics · see Rift damage</span></div>';
    } else if (state.layer === "lip" || state.layer === "lipheat") {
      const heat = state.layer === "lipheat";
      legend.innerHTML = '<div class="gradient" style="background:linear-gradient(90deg,#345966,#d4b26b,#d34b3d)"></div><div class="legend-labels"><span>0</span><span>' + (heat ? 'CURRENT THERMAL SUPPORT' : 'CUMULATIVE EMPLACEMENT EQUIVALENT') + '</span><span>' + (heat ? '800+ m' : '5+ km') + '</span></div>';
    } else if (state.layer === "thickness") {
      legend.innerHTML = '<div class="gradient" style="background:linear-gradient(90deg,#224b69,#c5b881,#ae6154)"></div><div class="legend-labels"><span>0 km</span><span>CRUST THICKNESS</span><span>70+ km</span></div>';
    } else if (state.layer === "basins") {
      legend.innerHTML = '<div class="gradient" style="background:linear-gradient(90deg,#b8beb1,#6695b8,#313b77)"></div><div class="legend-labels"><span>0 m</span><span>FORELAND DEFLECTION</span><span>2,000+ m</span></div>';
    } else if (state.layer === "damage") {
      legend.innerHTML = '<div class="gradient" style="background:linear-gradient(90deg,#345966,#d4b26b,#d34b3d)"></div><div class="legend-labels"><span>0 · INTACT</span><span>RIFT DAMAGE</span><span>1 · HIGH DAMAGE</span></div>';
    } else if (state.layer === "weakness") {
      legend.innerHTML = '<div class="gradient" style="background:linear-gradient(90deg,#d34b3d,#d4b26b,#345966)"></div><div class="legend-labels"><span>0 · WEAKER</span><span>RELATIVE STRENGTH</span><span>4+ · STRONGER</span></div>';
    } else if (state.layer === "deformation") {
      legend.innerHTML = '<div class="gradient" style="background:linear-gradient(90deg,#345966,#d4b26b,#d34b3d)"></div><div class="legend-labels"><span>RIGID INTERIOR</span><span>BELT PARTICIPATION</span><span>FULLY DEFORMING</span></div><div class="legend-swatches"><span><i style="background:#59636c"></i>Ocean · unavailable</span></div>';
    } else if (state.layer === "strain") {
      legend.innerHTML = '<div class="gradient" style="background:linear-gradient(90deg,#5274ad,#d9d7bd,#d34b3d)"></div><div class="legend-labels"><span>−10% · SHORTENING</span><span>0 · AREA CHANGE THIS STEP</span><span>+10% · STRETCHING</span></div><div class="legend-swatches"><span><i style="background:#59636c"></i>Ocean · unavailable</span></div>';
    } else if (state.layer === "refinement") {
      legend.innerHTML = '<div class="gradient" style="background:linear-gradient(90deg,#345966,#d4b26b,#d34b3d)"></div><div class="legend-labels"><span>STARTING DETAIL</span><span>REFINEMENT LEVEL</span><span>3 · FINEST</span></div><div class="legend-swatches"><span><i style="background:#59636c"></i>Ocean · unavailable</span></div>';
    } else if (state.layer === "age") {
      legend.innerHTML = '<div class="gradient" style="background:linear-gradient(90deg,#e3ac68,#71b1ac,#295981)"></div><div class="legend-labels"><span>NEW OCEAN</span><span>100 Myr</span><span>200+ Myr</span></div>';
    } else if (state.layer === "plate") {
      const description = hasDomains(state.frame) ? 'Each color represents a connected domain · some domains share plate motion' : 'Each color represents a plate · connected domains were not recorded';
      legend.innerHTML = '<div class="legend-swatches"><span><i style="background:#ad9463"></i><i style="background:#635fad"></i><i style="background:#5fad84"></i>' + description + '</span></div>';
    } else {
      legend.innerHTML = '<div class="gradient"></div><div class="legend-labels"><span>−11,000 m</span><span>SEA LEVEL</span><span>9,000 m</span></div>';
    }
  }

  function updateStats() {
    updateLipHistory();
    updateFoundationLayers();
    updateTrenchHistory();
    updateRiftHistory();
    updateSpreadingHistory();
    const data = state.mode === "initial" ? state.initial : state.frame;
    const firstLabel = document.querySelector(".metrics article .eyebrow");
    firstLabel.textContent = state.mode === "initial" ? "CONTINENTAL AREA" : "LAND ABOVE SEA";
    if (!data) return;
    const w = data.width, h = data.height;
    let land = 0, total = 0, peak = -Infinity, oceanAge = 0, ocean = 0;
    const plates = new Set(), domains = new Set();
    const recordedDomains = state.mode === "history" && hasDomains(data);
    for (let y = 0; y < h; y++) {
      const area = Math.max(.001, Math.cos(((y + .5) / h - .5) * Math.PI));
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        total += area;
        if (state.mode === "initial") { if (data.crust[i]) land += area; }
        else {
          if (data.elevation[i] >= 0) land += area;
          if (data.elevation[i] > peak) peak = data.elevation[i];
          plates.add(data.plate[i]);
          if (recordedDomains && data.domain[i] >= 0) domains.add(data.domain[i]);
          if (data.crust[i] === 0) { oceanAge += data.age[i] * area; ocean += area; }
        }
      }
    }
    $("stat-land").textContent = `${(100 * land / total).toFixed(1)}%`;
    $("stat-peak").textContent = state.mode === "initial" ? "—" : Math.round(peak).toLocaleString();
    $("stat-plates").textContent = state.mode === "initial" ? "—" : recordedDomains ? domains.size : plates.size;
    $("stat-plates-label").textContent = recordedDomains ? "SURFACE DOMAINS" : "ACTIVE PLATES";
    $("stat-plates-note").textContent = recordedDomains ? `${data.stats?.active_plates ?? plates.size} moving plates · shared motion possible` : "moving plates · domain identities not recorded";
    $("stat-age").textContent = state.mode === "initial" || !ocean ? "—" : (oceanAge / ocean).toFixed(1);
  }

  const eventName = (event) => String(event.type || "Tectonic event").replaceAll("_", " ");
  const formatTime = (time) => Number(time).toLocaleString(undefined, { maximumFractionDigits: 2 });
  const formatMetres = (value) => Number.isFinite(value) ? `${Math.round(value).toLocaleString()} m` : "Not recorded";

  function hasGridField(data, field) {
    return data && Number.isSafeInteger(data.width * data.height) && data.width * data.height > 0
      && (Array.isArray(data[field]) || ArrayBuffer.isView(data[field])) && data[field].length === data.width * data.height;
  }

  function updateFoundationLayers() {
    let changed = false;
    for (const [name, layer] of Object.entries(foundationLayers)) {
      const available = state.mode === "history" && hasGridField(state.frame, layer.field);
      $(layer.control).hidden = !available; $(layer.control).disabled = !available;
      if (!available && state.layer === name) { state.layer = state.mode === "initial" ? "crust" : "elevation"; changed = true; }
    }
    if (changed) {
      for (const button of document.querySelectorAll("[data-layer]")) {
        button.classList.toggle("active", button.dataset.layer === state.layer);
        button.setAttribute("aria-pressed", String(button.dataset.layer === state.layer));
      }
      updateLegend();
    }
  }

  function foundationColor(layer, value) {
    if (!Number.isFinite(value) || (value < 0 && layer !== "strain")) return [89, 99, 108];
    const stops = layer === "strain" ? [[82, 116, 173], [217, 215, 189], [211, 75, 61]]
      : ["damage", "deformation", "refinement", "lip", "lipheat"].includes(layer) ? [[52, 89, 102], [212, 178, 107], [211, 75, 61]]
      : layer === "weakness" ? [[211, 75, 61], [212, 178, 107], [52, 89, 102]]
      : layer === "thickness" ? [[34, 75, 105], [197, 184, 129], [174, 97, 84]] : [[184, 190, 177], [102, 149, 184], [49, 59, 119]];
    const t = Math.max(0, Math.min(1, layer === "strain" ? (value + 10) / 20 : value / ({ thickness: 70, basins: 2000, damage: 1, weakness: 4, refinement: 3, lip: 5, lipheat: 800 }[layer] || 1)));
    const a = t < .5 ? 0 : 1, f = t < .5 ? t * 2 : (t - .5) * 2;
    return stops[a].map((v, i) => Math.round(v + (stops[a + 1][i] - v) * f));
  }

  function foundationAccounting(point) {
    return point?.structure_version >= 1 || ["denudation_m", "rebound_m", "thermal_subsidence_m", "foreland_subsidence_m"].some((key) => Number.isFinite(point?.[key]));
  }

  function updateFoundationInspection(point) {
    const panel = $("inspection-structure"), values = $("inspection-structure-values");
    const fields = [["crustal_thickness_km", "Crust thickness", "km"], ["crustal_root_km", "Crustal root", "km"],
      ["rift_thermal_support_m", "Current rift thermal support", "m"], ["rift_cooling_age_myr", "Rift cooling age", "Myr"],
      ["foreland_deflection_m", "Current foreland deflection", "m"], ["denudation_m", "Cumulative denudation (gross)", "m"],
      ["rebound_m", "Cumulative rebound · included in net erosion", "m"],
      ["thermal_subsidence_m", "Cooling subsidence · included in tectonic lowering", "m"],
      ["foreland_subsidence_m", "Foreland subsidence · included in tectonic lowering", "m"]];
    values.replaceChildren();
    for (const [key, label, unit] of fields) if (Number.isFinite(point?.[key])) {
      const item = document.createElement("span");
      const value = key === "rift_cooling_age_myr" && point[key] < 0 ? "No recorded cooling episode"
        : `${point[key].toLocaleString(undefined, { maximumFractionDigits: unit === "m" ? 0 : 2 })} ${unit}`;
      item.textContent = `${label}: ${value}`; values.append(item);
    }
    panel.hidden = state.mode !== "history" || !values.children.length;
    $("inspection-structure-note").textContent = panel.hidden ? "" : "Thickness, root, thermal support and deflection are current structural estimates. Denudation and rebound explain the net erosion counter: rebound is already included there and must not be added to cumulative uplift. Cooling and foreland subsidence are already included in tectonic lowering.";
  }

  function trenchMilestones(system, events = []) {
    const combined = (system.history || []).filter((row) => Number.isFinite(row.time_myr) && typeof row.phase === "string")
      .map((row) => ({ type: `trench_${row.phase}`, time_myr: row.time_myr, description: String(row.reason || "").replaceAll("_", " "), details: { trench_id: system.id, episode: row.episode } }));
    for (const event of events) if (event.type?.startsWith("trench_") && event.details?.trench_id === system.id && Number.isFinite(event.time_myr)) combined.push(event);
    const unique = new Map();
    for (const event of combined) unique.set(`${event.time_myr}:${event.type}:${event.details?.episode ?? ""}`, event);
    return [...unique.values()].sort((a, b) => a.time_myr - b.time_myr || String(a.type).localeCompare(String(b.type)));
  }

  function updateTrenchHistory() {
    const panel = $("trench-history"), select = $("trench-select"), content = $("trench-detail"), data = state.frame;
    const recorded = state.mode === "history" && Array.isArray(data?.trench_systems);
    panel.hidden = !recorded; select.replaceChildren(); content.replaceChildren();
    $("trench-count").textContent = ""; $("trench-description").textContent = "";
    if (!recorded) { trenchReview.selectedId = null; return; }
    if (trenchReview.runId !== state.runId) { trenchReview.runId = state.runId; trenchReview.selectedId = null; }
    const systems = data.trench_systems.filter((row) => Number.isSafeInteger(row?.id) && row.id > 0).slice().sort((a, b) => a.id - b.id);
    const phase = (value) => ({ initiating: "Initiating", mature: "Mature", quiet: "Quiet", shutdown: "Shut down", joined: "Continues in another trench" }[value] || "Phase not recorded");
    $("trench-count").textContent = `${systems.filter((row) => row.phase === "mature" || row.phase === "initiating").length} ACTIVE / ${systems.length} RECORDED`;
    $("trench-description").textContent = systems.length ? `Local subduction systems recorded at ${formatTime(data.time_myr)} Myr. Maturity is a model indicator; phase buttons visit recorded history, with later events faded.` : "No local subduction systems recorded at this saved time.";
    select.disabled = !systems.length;
    if (!systems.some((row) => row.id === trenchReview.selectedId)) trenchReview.selectedId = (systems.find((row) => row.phase === "mature") || systems.find((row) => row.phase === "initiating") || systems[0])?.id ?? null;
    for (const row of systems) select.add(new Option(`Trench ${row.id} · ${phase(row.phase)}`, String(row.id)));
    select.value = trenchReview.selectedId == null ? "" : String(trenchReview.selectedId);
    const row = systems.find((entry) => entry.id === trenchReview.selectedId);
    if (!row) return;
    const names = new Map((data.plates || []).map((plate) => [plate.uid, plate.name]));
    const owner = (uid) => uid == null ? "Not recorded" : `${names.get(uid) || "Plate"} (UID ${uid})`;
    const identity = document.createElement("p"); identity.className = "hint";
    identity.textContent = `Downgoing: ${owner(row.downgoing_plate_uid)} · Overriding: ${owner(row.overriding_plate_uid)}`; content.append(identity);
    const quantity = (value, unit) => Number.isFinite(value) ? `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${unit}` : "Not recorded";
    const figures = document.createElement("div"); figures.className = "inspection-current";
    for (const value of [`Phase: ${phase(row.phase)}`, `Episode: ${Number.isSafeInteger(row.episode) ? row.episode : "Not recorded"}`,
      `Maturity: ${Number.isFinite(row.maturity) ? `${Math.round(row.maturity * 100)}%` : "Not recorded"}`,
      `Length: ${quantity(row.length_km, "km")}`, `Active this episode: ${quantity(row.active_myr, "Myr")}`,
      `Shortening this episode: ${quantity(row.shortening_km, "km")}`]) {
      const item = document.createElement("span"); item.textContent = value; figures.append(item);
    }
    content.append(figures);
    const lineage = [["Separated from", row.parent_trench_id], ["Preceding polarity", row.predecessor_trench_id], ["Continues as", row.successor_trench_id]].filter(([, id]) => Number.isSafeInteger(id));
    if (lineage.length) { const note = document.createElement("p"); note.className = "hint"; note.textContent = lineage.map(([label, id]) => `${label}: trench ${id}`).join(" · "); content.append(note); }
    const milestones = trenchMilestones(row, state.record?.events || data.events || []);
    const list = document.createElement("div"); list.className = "event-list";
    for (const event of milestones) list.append(eventButton(event));
    if (!milestones.length) { const note = document.createElement("p"); note.className = "hint"; note.textContent = "No recorded phase milestones for this system."; list.append(note); }
    content.append(list);
  }

  function riftMilestones(system, events = []) {
    const types = new Set(["rift_initiating", "rift_active", "rift_failed", "rift_reactivated", "rift_breakthrough"]);
    const phaseType = { incipient: "rift_initiating", active: "rift_active", failed: "rift_failed", broken_through: "rift_breakthrough", reactivated: "rift_reactivated" };
    const combined = (Array.isArray(system.history) ? system.history : []).filter((row) => Number.isFinite(row?.time_myr) && phaseType[row.phase])
      .map((row) => ({ type: phaseType[row.phase], time_myr: row.time_myr, description: String(row.reason || "").replaceAll("_", " "), details: { rift_system_id: system.id } }));
    for (const event of events) if (types.has(event.type) && event.details?.rift_system_id === system.id && Number.isFinite(event.time_myr)) combined.push(event);
    const unique = new Map();
    for (const event of combined) unique.set(`${event.time_myr}:${event.type === "rift_reactivated" ? "rift_active" : event.type}`, event);
    return [...unique.values()].sort((a, b) => a.time_myr - b.time_myr || String(a.type).localeCompare(String(b.type)));
  }

  function updateRiftHistory() {
    const panel = $("rift-history"), select = $("rift-select"), content = $("rift-detail"), data = state.frame;
    const recorded = state.mode === "history" && Array.isArray(data?.rift_systems);
    panel.hidden = !recorded; select.replaceChildren(); content.replaceChildren(); select.disabled = true;
    $("rift-count").textContent = ""; $("rift-description").textContent = "";
    $("rift-mechanics-detail").hidden = true; $("rift-mechanics-status").textContent = "";
    updateDeformationHistory();
    if (!recorded) { riftReview.selectedId = null; return; }
    if (riftReview.runId !== state.runId) { riftReview.runId = state.runId; riftReview.selectedId = null; }
    const systems = data.rift_systems.filter((row) => Number.isSafeInteger(row?.id) && row.id > 0).slice().sort((a, b) => a.id - b.id);
    const phase = (value) => ({ incipient: "Incipient", active: "Actively stretching", failed: "Failed / dormant", broken_through: "Broken through" }[value] || "Phase not recorded");
    $("rift-count").textContent = `${systems.filter((row) => ["incipient", "active"].includes(row.phase)).length} DEVELOPING / ${systems.length} RECORDED`;
    $("rift-description").textContent = systems.length ? `Local rift systems recorded at ${formatTime(data.time_myr)} Myr. Stretching can stall, reactivate or progress to continental breakup. Damage is a relative model indicator; stretching is cumulative distance, not current basin width.` : "No local continental rift systems recorded at this saved time.";
    const mechanics = data.rift_mechanics;
    if (mechanics && typeof mechanics === "object" && !Array.isArray(mechanics)) {
      const count = (value) => Number.isSafeInteger(value) && value >= 0 ? value.toLocaleString() : "Not recorded";
      $("rift-mechanics-detail").hidden = false;
      $("rift-mechanics-status").textContent = `${count(mechanics.mesh_nodes)} nodes · ${count(mechanics.mesh_edges)} connections · requested budget ${count(mechanics.target_nodes)} nodes. ${mechanics.converged === true ? "Approximate solve converged" : mechanics.converged === false ? "Approximate solve did not converge" : "Convergence not recorded"}; ${count(mechanics.solver_iterations)} iterations${Number.isFinite(mechanics.residual) ? `, residual ${mechanics.residual.toExponential(2)}` : ""}. This mechanical mesh is independent of the map and terrain grids.`;
    }
    select.disabled = !systems.length;
    if (!systems.some((row) => row.id === riftReview.selectedId)) riftReview.selectedId = (systems.find((row) => row.phase === "active") || systems.find((row) => row.phase === "incipient") || systems[0])?.id ?? null;
    for (const row of systems) select.add(new Option(`Rift system ${row.id} · ${phase(row.phase)}`, String(row.id)));
    select.value = riftReview.selectedId == null ? "" : String(riftReview.selectedId);
    const row = systems.find((entry) => entry.id === riftReview.selectedId);
    if (!row) return;
    const names = new Map((data.plates || []).map((plate) => [plate.uid, plate.name]));
    const owner = (uid) => uid == null ? "Not recorded" : `${names.get(uid) || "Plate"} (UID ${uid})`;
    const identity = document.createElement("p"); identity.className = "hint";
    identity.textContent = `Parent: ${owner(row.plate_uid)}${row.new_plate_uid == null ? "" : ` · New plate: ${owner(row.new_plate_uid)}`}`; content.append(identity);
    if (Number.isSafeInteger(row.source_rift_id) && row.source_rift_id > 0) {
      const scar = document.createElement("p"); scar.className = "hint";
      scar.textContent = `Linked rift scar: ${row.source_rift_id}. This separate scar identity links to the regional rift-inversion history.`; content.append(scar);
    }
    const quantity = (value, unit = "") => Number.isFinite(value) ? `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}${unit ? ` ${unit}` : ""}` : "Not recorded";
    const figures = document.createElement("div"); figures.className = "inspection-current";
    for (const value of [`Phase: ${phase(row.phase)}`, `Cumulative stretching: ${quantity(row.extension_km, "km")}`,
      `Peak damage: ${Number.isFinite(row.peak_damage) ? `${Math.round(row.peak_damage * 100)}%` : "Not recorded"}`,
      `Mean relative strength: ${quantity(row.mean_strength)}`, `Started: ${quantity(row.started_myr, "Myr")}`, `Last active: ${quantity(row.last_active_myr, "Myr")}`]) {
      const item = document.createElement("span"); item.textContent = value; figures.append(item);
    }
    content.append(figures);
    const milestones = riftMilestones(row, state.record?.events || data.events || []);
    const list = document.createElement("div"); list.className = "event-list";
    for (const event of milestones) list.append(eventButton(event));
    if (!milestones.length) { const note = document.createElement("p"); note.className = "hint"; note.textContent = "No recorded phase milestones for this system."; list.append(note); }
    content.append(list);
  }

  function updateDeformationHistory() {
    const panel = $("deformation-detail"), text = $("deformation-status"), data = state.frame;
    const deformation = data?.deformation_diagnostics, adaptation = data?.adaptivity_diagnostics;
    panel.hidden = state.mode !== "history" || (!deformation && !adaptation);
    text.textContent = "";
    if (panel.hidden) return;
    const entries = [], count = (value) => Number.isFinite(value) ? Math.round(value).toLocaleString() : "Not recorded";
    const faces = adaptation?.output_faces ?? adaptation?.material_faces ?? data.mesh_diagnostics?.material_faces;
    if (Number.isFinite(faces)) entries.push(`${count(faces)} material triangles`);
    if (Number.isFinite(adaptation?.face_budget)) entries.push(`limit ${count(adaptation.face_budget)}`);
    if (adaptation?.refinement) entries.push(`refinement ${count(adaptation.refinement.old_faces)} → ${count(adaptation.refinement.new_faces)} triangles`);
    if (adaptation?.coarsening) entries.push(`coarsening ${count(adaptation.coarsening.old_faces)} → ${count(adaptation.coarsening.new_faces)} triangles`);
    const deferred = adaptation?.refinement_request?.budget_deferred_edges;
    if (Number.isFinite(deferred) && deferred > 0) entries.push(`${count(deferred)} refinements postponed by the limit`);
    if (adaptation?.budget_exceeded_by_existing_material) entries.push("Existing crust exceeds the limit; further refinement is paused");
    if (Number.isFinite(deformation?.limited_faces)) entries.push(`${count(deformation.limited_faces)} faces limited to preserve their shape`);
    if (Number.isFinite(deformation?.accepted_residual_fraction)) entries.push(`${(100 * deformation.accepted_residual_fraction).toFixed(1)}% of requested deformation accepted`);
    text.textContent = `${entries.join(" · ")}${entries.length ? ". " : ""}Area strain shows the actual change during the last time step; zero can also occur in a deforming belt with no change in area.`;
  }

  function updateSpreadingHistory() {
    const panel = $("spreading-history"), figures = $("spreading-figures"), detail = $("spreading-budget");
    const data = state.frame, diagnostic = data?.spreading_diagnostics;
    const recorded = state.mode === "history" && diagnostic && typeof diagnostic === "object" && !Array.isArray(diagnostic)
      && ["active_segments", "reconstructed_cells", "generated_area_km2", "mean_full_rate_cm_yr", "mean_axis_speed_cm_yr"].some((key) => Object.hasOwn(diagnostic, key));
    panel.hidden = !recorded;
    figures.replaceChildren();
    detail.textContent = "";
    $("spreading-epoch").textContent = "";
    $("spreading-description").textContent = "";
    if (!recorded) return;
    const quantity = (value, unit, digits = 2) => Number.isFinite(value) && value >= 0
      ? `${value.toLocaleString(undefined, { maximumFractionDigits: digits })}${unit ? ` ${unit}` : ""}` : "Not recorded";
    const count = (value) => Number.isSafeInteger(value) && value >= 0 ? value.toLocaleString() : "Not recorded";
    const initial = data.time_myr === 0;
    $("spreading-epoch").textContent = Number.isFinite(data.time_myr) ? `${formatTime(data.time_myr)} Myr` : "Saved time not recorded";
    $("spreading-description").textContent = initial ? "Initial saved state; no elapsed integration step yet." : "These figures describe the last integration step at this saved time, not the whole interval between snapshots. Rates are length-weighted means.";
    for (const [label, value] of [["Mean full opening", quantity(diagnostic.mean_full_rate_cm_yr, "cm/yr")],
      ["Mean axis drift", quantity(diagnostic.mean_axis_speed_cm_yr, "cm/yr")],
      ["Ocean added this step", quantity(diagnostic.generated_area_km2, "km²")],
      ["Eligible spreading segments", count(diagnostic.active_segments)]]) {
      const item = document.createElement("span"); item.textContent = `${label}: ${value}`; figures.append(item);
    }
    detail.textContent = `Reconstructed cells: ${count(diagnostic.reconstructed_cells)} · Side P added: ${quantity(diagnostic.side_p_area_km2, "km²")} · Side Q added: ${quantity(diagnostic.side_q_area_km2, "km²")}. P and Q group the two local sides of each segment, not two global plates. Areas are realized new-ocean reference area within mixed cells. Eligible segments can contribute no area where nearby junctions block reconstruction.`;
  }

  function nearestFrame(time) {
    let best = 0, distance = Infinity;
    state.frames.forEach((frame, index) => { const delta = Math.abs(Number(frame.time_myr) - time); if (delta < distance) { distance = delta; best = index; } });
    return best;
  }

  function visitTime(time) { if (!state.frames.length || state.loadingRun) return; stopPlayback(); showFrame(nearestFrame(time)); }

  function eventButton(event) {
    const row = document.createElement("button"); row.className = "event-item"; row.type = "button";
    row.classList.toggle("future-event", Number(event.time_myr) > Number(state.frame?.time_myr || 0));
    row.title = `Visit the saved frame nearest ${formatTime(event.time_myr || 0)} Myr`;
    const time = document.createElement("span"); time.className = "event-time"; time.textContent = `${formatTime(event.time_myr || 0)} Myr`;
    const body = document.createElement("span"); body.className = "event-body";
    const title = document.createElement("strong"); title.textContent = eventName(event);
    const detail = document.createElement("span");
    const description = String(event.description || event.message || ""), systemId = event.details?.rift_system_id;
    const oldPrefix = `Continental rift ${systemId}:`;
    detail.textContent = Number.isSafeInteger(systemId) && systemId > 0 && description.startsWith(oldPrefix)
      ? `Rift system ${systemId}:${description.slice(oldPrefix.length)}` : description;
    body.append(title, detail); row.append(time, body);
    row.addEventListener("click", () => visitTime(Number(event.time_myr || 0)));
    return row;
  }

  function updateEvents() {
    updateBackArcHistory();
    updateTrenchHistory();
    updateRiftHistory();
    const list = $("event-list"), scroll = list.scrollTop;
    list.replaceChildren();
    const events = state.record?.events || [];
    const typeSelect = $("event-type"), previousType = typeSelect.value;
    const types = [...new Set(events.map((event) => event.type || ""))].sort();
    if (typeSelect.dataset.types !== JSON.stringify(types)) {
      typeSelect.replaceChildren(new Option("All event types", ""));
      for (const type of types) typeSelect.add(new Option(type.replaceAll("_", " "), type));
      if (types.includes(previousType)) typeSelect.value = previousType;
      typeSelect.dataset.types = JSON.stringify(types);
    }
    const search = $("event-search").value.trim().toLowerCase(), type = typeSelect.value;
    const filtered = events.filter((event) => (!type || event.type === type) && (!search || `${eventName(event)} ${event.description || event.message || ""} ${(event.plate_ids || []).join(" ")}`.toLowerCase().includes(search)));
    $("event-count").textContent = `${filtered.length} / ${events.length} EVENTS`;
    $("record-note").textContent = state.recordFallback ? "Older engine: showing the event record from the latest saved snapshot. Events remain visible when you scrub." : "The whole experiment stays visible as you scrub. Faded events occur after the displayed time; click any event to visit it.";
    if (!filtered.length) {
      const empty = document.createElement("p"); empty.className = "empty-events";
      empty.textContent = events.length ? "No events match these filters." : state.frames.length ? "No discrete events recorded yet. Motion and surface evolution are preserved in the saved frames." : "Run an experiment to record its tectonic history.";
      list.append(empty);
    }
    for (const event of filtered.slice().sort((a, b) => Number(a.time_myr) - Number(b.time_myr))) list.append(eventButton(event));
    list.scrollTop = scroll;
    updateEventTrack(filtered);
  }

  function backArcMilestones(basin, events) {
    const types = new Set(["backarc_loading", "backarc_rifting", "backarc_spreading", "backarc_quiet", "backarc_closed"]);
    if (!Number.isSafeInteger(basin?.id)) return [];
    return (events || []).filter((event) => types.has(event.type) && (event.details?.backarc_id ?? event.backarc_id) === basin.id && Number.isFinite(event.time_myr))
      .slice().sort((a, b) => a.time_myr - b.time_myr);
  }

  function updateBackArcHistory() {
    updateCollisionHistory();
    updateAuthoredHistory();
    const panel = $("backarc-history"), list = $("backarc-basins"), data = state.frame;
    const recorded = state.mode === "history" && Array.isArray(data?.backarc_basins);
    panel.hidden = !recorded;
    if (!recorded) { list.replaceChildren(); return; }
    if (backarcReview.runId !== state.runId) { backarcReview.runId = state.runId; backarcReview.expanded.clear(); backarcReview.collapsed.clear(); }
    const basins = data.backarc_basins.filter((row) => row && Number.isSafeInteger(row.id)).slice().sort((a, b) => a.id - b.id);
    const defaultBasin = basins.find((basin) => basin.arc_plate_uid != null) || basins[0];
    $("backarc-count").textContent = `${basins.length} RECORDED`;
    $("backarc-description").textContent = basins.length ? `Saved basin states at ${formatTime(data.time_myr)} Myr. Loading can precede rupture; only measured extension counts as opening. Expand a basin to review its recorded milestones.` : "No back-arc basins recorded at this saved time.";
    list.replaceChildren();
    const phases = { loading: "Loading", rifting: "Rifting", spreading: "Spreading", quiet: "Quiet", closed: "Closed" };
    const plates = new Map((data.plates || []).filter((plate) => plate.uid != null).map((plate) => [plate.uid, plate.name]));
    const plateName = (uid) => uid == null ? "Not recorded" : `${plates.get(uid) || "Plate"} (UID ${uid})`;
    const quantity = (value, unit) => Number.isFinite(value) ? `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${unit}` : "Not recorded";
    const events = state.record?.events || data.events || [];
    basins.forEach((basin) => {
      const details = document.createElement("details"); details.className = "inspection-details";
      details.open = backarcReview.expanded.has(basin.id) || (basin.id === defaultBasin.id && !backarcReview.collapsed.has(basin.id));
      details.addEventListener("toggle", () => { if (details.open) { backarcReview.expanded.add(basin.id); backarcReview.collapsed.delete(basin.id); } else { backarcReview.collapsed.add(basin.id); backarcReview.expanded.delete(basin.id); } });
      const summary = document.createElement("summary"); summary.textContent = `Basin ${basin.id} · ${phases[basin.phase] || "Phase not recorded"}`; details.append(summary);
      const identities = document.createElement("p"); identities.className = "hint";
      const arc = basin.arc_plate_uid === null ? "not separated" : plateName(basin.arc_plate_uid);
      identities.textContent = `Parent: ${plateName(basin.parent_plate_uid)} · Arc: ${arc} · Downgoing: ${plateName(basin.downgoing_plate_uid)}`; details.append(identities);
      const figures = document.createElement("div"); figures.className = "inspection-current";
      for (const [label, value] of [["Accumulated extension", quantity(basin.opening_km, "km")], ["Current opening rate", quantity(basin.extension_rate_km_myr, "km/Myr")], ["Loading", quantity(basin.loading_km, "km")]]) {
        const item = document.createElement("span"); item.textContent = `${label}: ${value}`; figures.append(item);
      }
      if (basin.arc_plate_uid === null && Number.isFinite(basin.loading_rate_km_myr)) {
        const item = document.createElement("span"); item.textContent = `Back-arc loading rate: ${quantity(basin.loading_rate_km_myr, "km/Myr")}`; figures.append(item);
      }
      details.append(figures);
      const dates = document.createElement("p"); dates.className = "hint";
      const milestones = [`First recorded: ${quantity(basin.started_myr, "Myr")}`, `Last active: ${quantity(basin.last_active_myr, "Myr")}`];
      if (Number.isFinite(basin.rupture_myr)) milestones.push(`Rupture: ${quantity(basin.rupture_myr, "Myr")}`);
      const center = basin.center;
      if (Array.isArray(center) && center.length === 3 && center.every(Number.isFinite) && Math.hypot(...center) > 1e-12) {
        const norm = Math.hypot(...center), lon = Math.atan2(center[1], center[0]) * 180 / Math.PI, lat = Math.asin(Math.max(-1, Math.min(1, center[2] / norm))) * 180 / Math.PI;
        milestones.push(`Center: ${Math.abs(lat).toFixed(1)}° ${lat < 0 ? "S" : "N"}, ${Math.abs(lon).toFixed(1)}° ${lon < 0 ? "W" : "E"}`);
      }
      dates.textContent = milestones.join(" · "); details.append(dates);
      const note = document.createElement("p"); note.className = "hint"; note.textContent = "Accumulated extension records opening over time, not the present basin width. Event buttons visit saved milestones; later events are faded."; details.append(note);
      const catalogue = document.createElement("div"); catalogue.className = "event-list";
      const records = backArcMilestones(basin, events);
      for (const event of records) catalogue.append(eventButton(event));
      if (!records.length) { const empty = document.createElement("p"); empty.className = "hint"; empty.textContent = "No discrete milestones recorded for this basin."; catalogue.append(empty); }
      details.append(catalogue); list.append(details);
    });
  }

  function updateEventTrack(events) {
    const track = $("event-track"); track.replaceChildren();
    const last = Number(state.frames.at(-1)?.time_myr || 0);
    if (!last) return;
    const groups = new Map();
    for (const event of events) { const index = nearestFrame(event.time_myr); if (!groups.has(index)) groups.set(index, []); groups.get(index).push(event); }
    for (const [index, items] of groups) {
      const mark = document.createElement("button"); mark.className = "event-mark";
      mark.style.left = `${state.frames.length > 1 ? index / (state.frames.length - 1) * 100 : 0}%`;
      mark.classList.toggle("future-event", Number(items[0].time_myr) > Number(state.frame?.time_myr || 0));
      mark.title = `${formatTime(state.frames[index].time_myr)} Myr · ${items.map(eventName).join(", ")}`;
      mark.setAttribute("aria-label", mark.title);
      mark.addEventListener("click", () => { stopPlayback(); showFrame(index); });
      track.append(mark);
    }
  }

  async function refreshRecord() {
    if (!state.runId || !state.frames.length) return;
    const run = state.runId, count = state.frames.length, ticket = ++state.recordRequest, pose = outputOrientation();
    state.recordCount = count;
    try {
      let record, fallback = false;
      try { record = await api(orientedPath(`/api/record?run_id=${encodeURIComponent(run)}`, pose)); }
      catch (error) {
        if (error.status !== 404) throw error;
        const latest = state.cache.get(frameCacheKey(count - 1, pose, run)) || await api(orientedPath(`/api/frame?index=${count - 1}`, pose));
        record = { run_id: run, frame_count: count, events: latest.events || [], history_version: 0 }; fallback = true;
      }
      if (run !== state.runId || ticket !== state.recordRequest || !sameOrientation(pose, outputOrientation())) return;
      state.record = record; state.recordFallback = fallback;
      updateEvents();
    } catch (error) {
      if (run !== state.runId || ticket !== state.recordRequest) return;
      state.recordCount = -1;
      $("record-note").textContent = `Could not read the event record: ${error.message}`;
    }
  }

  function clearInspection() {
    state.inspectionRequest++; state.inspection = null; state.inspectionPending = false;
    updateInspection(); render();
  }

  function resetHistory() {
    state.recordRequest++; state.record = null; state.recordCount = -1; state.recordFallback = false;
    state.followLatest = true; $("follow-live").checked = true;
    $("event-search").value = ""; $("event-type").value = "";
    clearInspection(); updateEvents();
  }

  // Great-circle distance on a sphere of Earth's mean radius. The simulation
  // itself carries RADIUS_KM = 6371 and its own areas are computed on that
  // sphere, so this measure is consistent with the model's own geometry rather
  // than an outside convention laid over it.
  const MEASURE_RADIUS_KM = 6371;

  function measureCoordinates(point) {
    const data = state.mode === "initial" ? state.initial : state.frame;
    if (!data) return null;
    // Prefer the exact click position when the picker supplied one. Falling back to
    // the cell centre keeps older call sites working, but at ~208 km per cell that
    // fallback is the difference between a real measurement and a rounded one.
    if (typeof point.u === "number" && typeof point.v === "number") {
      return { lon: point.u * 360 - 180, lat: 90 - point.v * 180 };
    }
    return { lon: (point.x + .5) / data.width * 360 - 180,
             lat: 90 - (point.y + .5) / data.height * 180 };
  }

  function formatLatLon(p) {
    return `${Math.abs(p.lat).toFixed(2)}° ${p.lat >= 0 ? "N" : "S"}, ${Math.abs(p.lon).toFixed(2)}° ${p.lon >= 0 ? "E" : "W"}`;
  }

  function greatCircleKm(a, b) {
    const rad = Math.PI / 180;
    const p1 = a.lat * rad, p2 = b.lat * rad, dp = (b.lat - a.lat) * rad, dl = (b.lon - a.lon) * rad;
    // Haversine: numerically stable for the small separations a click pair can
    // produce, where the spherical law of cosines loses precision.
    const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
    return 2 * MEASURE_RADIUS_KM * Math.asin(Math.min(1, Math.sqrt(h)));
  }

  function initialBearing(a, b) {
    const rad = Math.PI / 180;
    const p1 = a.lat * rad, p2 = b.lat * rad, dl = (b.lon - a.lon) * rad;
    const deg = Math.atan2(Math.sin(dl) * Math.cos(p2),
      Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl)) / rad;
    return (deg + 360) % 360;
  }

  function updateMeasureReadout() {
    const readout = $("measure-readout"), clear = $("measure-clear");
    const pts = state.measure.points;
    clear.hidden = !state.measure.active || pts.length === 0;
    if (!state.measure.active) { readout.hidden = true; return; }
    readout.hidden = false;
    if (pts.length === 0) { readout.textContent = "Measure: click the first point."; return; }
    if (pts.length === 1) { readout.textContent = `From ${formatLatLon(pts[0])} — click the second point.`; return; }
    const km = greatCircleKm(pts[0], pts[1]);
    const bearing = initialBearing(pts[0], pts[1]);
    const arc = km / (2 * Math.PI * MEASURE_RADIUS_KM) * 360;
    readout.textContent =
      `${formatLatLon(pts[0])}  →  ${formatLatLon(pts[1])}\n` +
      `${km.toLocaleString(undefined, { maximumFractionDigits: 0 })} km  ·  ${arc.toFixed(2)}° of arc  ·  bearing ${bearing.toFixed(0)}°\n` +
      `Great circle on R = ${MEASURE_RADIUS_KM.toLocaleString()} km. Click again to start a new measurement.`;
  }

  function setMeasureActive(active) {
    state.measure.active = active;
    state.measure.points = [];
    $("measure-toggle").setAttribute("aria-pressed", active ? "true" : "false");
    $("measure-toggle").classList.toggle("active", active);
    if (!active) $("measure-readout").hidden = true;
    updateMeasureReadout();
  }

  function addMeasurePoint(point) {
    const coords = measureCoordinates(point);
    if (!coords) return;
    if (state.measure.points.length >= 2) state.measure.points = [];
    state.measure.points.push(coords);
    updateMeasureReadout();
  }

  async function inspectPoint(point) {
    // Measuring intercepts inspection so a click cannot both move the history
    // cursor and drop a measurement pin.
    if (state.measure.active) { addMeasurePoint(point); return; }
    if (state.mode !== "history" || !state.frame || state.loadingRun || state.frame.index !== state.currentIndex) return;
    await loadInspection(state.currentIndex, point.y * state.frame.width + point.x);
  }

  async function loadInspection(anchorFrame, cell) {
    stopPlayback();
    state.followLatest = false; $("follow-live").checked = false;
    const run = state.runId, frame = state.currentIndex, ticket = ++state.inspectionRequest, pose = outputOrientation();
    state.inspection = null; state.inspectionPending = true; updateInspection(); render();
    try {
      const result = await api(orientedPath(`/api/history?run_id=${encodeURIComponent(run)}&frame=${anchorFrame}&cell=${cell}`, pose));
      if (ticket !== state.inspectionRequest || run !== state.runId || frame !== state.currentIndex || state.mode !== "history" || !sameOrientation(pose, outputOrientation())) return;
      state.inspection = result; state.inspectionPending = false;
      updateInspection(); render();
    } catch (error) {
      if (ticket !== state.inspectionRequest || run !== state.runId || frame !== state.currentIndex) return;
      state.inspectionPending = false; updateInspection();
      $("inspector-description").textContent = error.status === 404 ? "Regional inspection needs the updated local engine. The recorded maps and whole-experiment event browser remain available." : `Could not read this region's history: ${error.message}`;
    }
  }

  function selectedHistoryPoint() {
    return state.inspection?.points.find((point) => point.index === state.currentIndex && Number.isFinite(point.elevation_m));
  }

  function riftEvidenceRows(points) {
    const seen = new Map(), rows = [];
    for (const point of points) {
      if (!Number.isSafeInteger(point.rift_id) || point.rift_id < 1 || !Number.isFinite(point.time_myr)) continue;
      const id = point.rift_id;
      if (!seen.has(id)) {
        seen.set(id, { extension: false, inversion: false });
        const birth = Number.isFinite(point.rift_birth_myr) && point.rift_birth_myr >= 0 ? ` · scar created at ${formatTime(point.rift_birth_myr)} Myr` : "";
        rows.push({ time_myr: point.time_myr, text: `Rift ${id} origin first recorded for this material${birth}. Scar creation alone does not record stretching.` });
      }
      const recorded = seen.get(id);
      if (!recorded.extension && Number.isFinite(point.rift_extension_m) && point.rift_extension_m > 0) {
        recorded.extension = true;
        rows.push({ time_myr: point.time_myr, text: `Rift ${id} stretching first present in saved material history: ${formatMetres(point.rift_extension_m)} of cumulative depression.` });
      }
      if (!recorded.inversion && Number.isFinite(point.inversion_uplift_m) && point.inversion_uplift_m > 0) {
        recorded.inversion = true;
        rows.push({ time_myr: point.time_myr, text: `Rift ${id} inversion first present in saved material history: ${formatMetres(point.inversion_uplift_m)} of cumulative uplift.` });
      }
    }
    return rows;
  }

  function updateInspection() {
    const history = state.inspection;
    updateFoundationInspection(state.mode === "history" && history ? selectedHistoryPoint() : null);
    $("clear-inspection").hidden = !history && !state.inspectionPending;
    $("refresh-inspection").hidden = !history || state.mode !== "history" || Number(history.points.at(-1)?.index ?? -1) >= state.frames.length - 1;
    $("inspection-content").hidden = !history || state.mode !== "history";
    if (state.mode === "initial") { $("inspector-title").textContent = "How did this place form?"; $("inspector-description").textContent = "Open recorded history, then click the map to inspect a region through time."; return; }
    if (!history) {
      $("inspector-title").textContent = state.inspectionPending ? "Reading this region's history…" : "How did this place form?";
      $("inspector-description").textContent = state.inspectionPending ? "Following the selected point across the saved record." : "Click the recorded map to follow a material marker where available, or examine a fixed geographic location.";
      return;
    }
    const material = history.mode === "material", point = selectedHistoryPoint();
    $("inspector-title").textContent = material ? "Follow this piece of crust" : "History at this location";
    let description = history.description || (material ? "The same material marker is followed as plates move and change ownership." : "This follows a fixed longitude and latitude. Different pieces of crust may pass through it.");
    if (material && Number.isFinite(history.selection?.marker_distance_km)) description += ` Selected marker is ${Math.round(history.selection.marker_distance_km).toLocaleString()} km from the clicked cell centre.`;
    if (history.points.length) description += ` Record available through ${formatTime(history.points.at(-1).time_myr)} Myr.`;
    $("inspector-description").textContent = description;
    $("inspection-mode").textContent = material ? `TRACKED MATERIAL · MARKER ${history.trace_id}` : "FIXED LOCATION · DIFFERENT CRUST MAY PASS THROUGH";
    const current = $("inspection-current"); current.replaceChildren();
    if (point) {
      const materialKind = material ? point.material_kind ?? point.crust : point.crust;
      const parts = [`${formatTime(point.time_myr)} Myr`, formatMetres(point.elevation_m), regionName(point), crustNames[materialKind] || "Crust", boundaryNames[point.boundary] || "No active boundary"];
      if (point.boundary === 0) parts[4] = "No active boundary";
      if (point.domain_uid != null) parts.push(`Motion: ${point.plate_name || `Plate ${point.plate_id}`}`);
      if (material && materialKind !== point.crust) parts.push(`Mapped cell: ${crustNames[point.crust] || "crust"}`);
      for (const value of parts) { const item = document.createElement("span"); item.textContent = value; current.append(item); }
    } else current.textContent = "This material is not present in the displayed frame. Its plotted history begins when it was recorded.";
    const processes = $("inspection-processes"); processes.replaceChildren();
    const structural = foundationAccounting(point);
    const fields = [["uplift_m", "Cumulative uplift"], ["extension_m", structural ? "Tectonic lowering" : "Extension loss"], ["erosion_m", structural ? "Net erosion lowering" : "Relief relaxation loss"], ["adjustment_m", "Other relief adjustment"], ["relief_m", "Current relief proxy"]];
    if (Number.isFinite(point?.ridge_uplift_m)) fields.push(["ridge_uplift_m", "Ridge volcanism · included above"]);
    if (Number.isFinite(point?.ridge_thermal_m)) fields.push(["ridge_thermal_m", "Current slab-window thermal uplift"]);
    if (Number.isSafeInteger(point?.rift_id) && point.rift_id >= 1) fields.push(["rift_id", "Rift origin"]);
    if (Number.isFinite(point?.rift_extension_m)) fields.push(["rift_extension_m", "Rift stretching · included above"]);
    if (Number.isFinite(point?.inversion_uplift_m)) fields.push(["inversion_uplift_m", "Rift inversion · included above"]);
    for (const [key, name] of fields) {
      const item = document.createElement("div"), label = document.createElement("span"), value = document.createElement("strong");
      label.textContent = name; value.textContent = key === "rift_id" ? `Rift ${point.rift_id}` : formatMetres(point?.[key]); item.append(label, value); processes.append(item);
    }
    const hasAccounting = material && history.points.some((p) => Number.isFinite(p.uplift_m));
    $("inspection-accounting").textContent = hasAccounting ? structural
      ? "Cumulative terms describe the model's relief proxy, not a complete surface-elevation budget. Tectonic lowering includes stretching, cooling and foreland subsidence. Net erosion lowering already includes rebound: positive lowers relief, negative raises it. Other adjustments and surface terms can also affect the map."
      : "Cumulative terms describe the model's relief proxy, not a complete surface-elevation budget. Relaxation loss is signed: a negative value raises depressed relief. Other adjustment includes clipping and arc resets. Bathymetry and other surface terms also affect the map."
      : "This run or location has no tracked material process totals. Elevation changes can include motion of different crust through a cell; they are not a direct uplift rate.";
    if (point?.suture != null) $("inspection-accounting").textContent += ` Suture memory: ${Number(point.suture).toFixed(2)} (model proxy).`;
    if (Number.isFinite(point?.ridge_uplift_m)) $("inspection-accounting").textContent += " Ridge-subduction volcanic uplift is already included in cumulative uplift; do not add it twice.";
    if (Number.isFinite(point?.ridge_thermal_m)) $("inspection-accounting").textContent += " Slab-window thermal uplift is a temporary offset additional to the relief proxy. It fades as the episode cools; it is not a cumulative total.";
    if (Number.isFinite(point?.rift_extension_m)) $("inspection-accounting").textContent += ` Rift stretching is observed depression already included in ${structural ? "tectonic lowering" : "extension loss"}.`;
    if (Number.isFinite(point?.inversion_uplift_m)) $("inspection-accounting").textContent += " Rift inversion is realized uplift already included in cumulative uplift; do not add it twice.";
    if (Number.isSafeInteger(point?.rift_id) && point.rift_id >= 1) {
      const birth = Number.isFinite(point.rift_birth_myr) && point.rift_birth_myr >= 0 ? ` at ${formatTime(point.rift_birth_myr)} Myr` : "";
      $("inspection-accounting").textContent += ` Rift ${point.rift_id} identifies the earliest recorded local structural scar, created${birth || " at an unrecorded time"}. A starting scar does not imply earlier stretching. Review its recorded sequence below.`;
    }
    drawElevationChart();
    const changes = $("inspection-changes"); changes.replaceChildren();
    let previous = null;
    for (const p of history.points) {
      if (!Number.isFinite(p.elevation_m)) continue;
      const identity = regionIdentity(p);
      if (!previous || identity !== regionIdentity(previous) || p.boundary !== previous.boundary || p.crust !== previous.crust) {
        const row = document.createElement("button"); row.className = "history-change";
        const label = [regionName(p), p.domain_uid != null ? `motion: ${p.plate_name || `Plate ${p.plate_id}`}` : null, crustNames[p.crust], p.boundary ? boundaryNames[p.boundary] : "No active boundary"].filter(Boolean).join(" · ");
        row.textContent = `${formatTime(p.time_myr)} Myr — ${label}`;
        row.addEventListener("click", () => visitTime(p.time_myr)); changes.append(row);
      }
      previous = p;
    }
    const riftEvidence = riftEvidenceRows(history.points);
    if (riftEvidence.length) { const label = document.createElement("p"); label.className = "hint"; label.textContent = "Recorded rift sequence · first saved evidence may be later than the process began."; changes.append(label); }
    for (const evidence of riftEvidence) {
      const row = document.createElement("button"); row.className = "history-change";
      row.textContent = `${formatTime(evidence.time_myr)} Myr — ${evidence.text}`;
      row.title = "Visit the saved frame containing this material evidence; first recorded evidence may be later than the process began.";
      row.addEventListener("click", () => visitTime(evidence.time_myr)); changes.append(row);
    }
    const related = $("inspection-events"); related.replaceChildren();
    const note = document.createElement("p"); note.className = "hint"; note.textContent = "These events involve this marker's host plates and can occur elsewhere on them. They do not establish the cause of local terrain.";
    if (!material) note.textContent = "These events involve plates occupying this location at recorded times; they can occur elsewhere on those plates.";
    related.append(note);
    for (const event of history.events || []) related.append(eventButton(event));
    if (!history.events?.length) { const empty = document.createElement("p"); empty.className = "hint"; empty.textContent = "No related discrete events recorded."; related.append(empty); }
  }

  function drawElevationChart() {
    const svg = $("elevation-history"); svg.replaceChildren();
    const points = state.inspection.points.filter((p) => Number.isFinite(p.elevation_m) && Number.isFinite(p.time_myr));
    if (!points.length) return;
    const maxTime = Math.max(1, Number(state.frames.at(-1)?.time_myr || points.at(-1).time_myr));
    let minimum = Math.min(0, ...points.map((p) => p.elevation_m)), maximum = Math.max(0, ...points.map((p) => p.elevation_m));
    const pad = Math.max(100, (maximum - minimum) * .1); minimum -= pad; maximum += pad;
    const x = (t) => 55 + t / maxTime * 571, y = (elevation) => 13 + (maximum - elevation) / (maximum - minimum) * 127;
    const element = (name, attributes, text) => { const el = document.createElementNS("http://www.w3.org/2000/svg", name); for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value); if (text != null) el.textContent = text; svg.append(el); return el; };
    for (const value of [maximum - pad, 0, minimum + pad].filter((v, i, a) => a.indexOf(v) === i)) {
      element("line", { x1: 55, x2: 626, y1: y(value), y2: y(value), class: value === 0 ? "chart-sea" : "chart-grid" });
      element("text", { x: 48, y: y(value) + 3, "text-anchor": "end", class: "chart-label" }, Math.round(value).toLocaleString());
    }
    for (const time of [0, maxTime / 2, maxTime]) element("text", { x: x(time), y: 161, "text-anchor": time === 0 ? "start" : time === maxTime ? "end" : "middle", class: "chart-label" }, `${formatTime(time)} Myr`);
    let path = "", previousIndex = null;
    for (const point of points) { path += `${previousIndex === point.index - 1 ? "L" : "M"}${x(point.time_myr).toFixed(2)},${y(point.elevation_m).toFixed(2)} `; previousIndex = point.index; }
    element("path", { d: path, class: "chart-elevation" });
    if (points.length === 1) element("circle", { cx: x(points[0].time_myr), cy: y(points[0].elevation_m), r: 3, class: "chart-point" });
    const time = Number(state.frame?.time_myr || 0), selected = selectedHistoryPoint();
    element("line", { x1: x(time), x2: x(time), y1: 9, y2: 144, class: "chart-current" });
    if (selected) element("circle", { cx: x(time), cy: y(selected.elevation_m), r: 4, class: "chart-point" });
    svg.dataset.maxTime = maxTime;
  }

  // CSS pixels, shared with the legend. The minimum and logarithmic scale reveal
  // slow plates without letting a fast remnant fill the map.
  function motionArrowLength(speed) {
    return Number.isFinite(speed) && speed > 0 ? 10 + 7 * Math.log1p(Math.min(30, speed) / .2) : 0;
  }

  function updateMotionLegend() {
    const legend = $("motion-legend");
    legend.hidden = state.mode === "initial" || !$("show-motion").checked;
    $("motion-fullscreen-legend").hidden = legend.hidden;
    if (legend.hidden || legend.innerHTML) return;
    const samples = [.1, .5, 1, 5, 30].map(speed => {
      const length = motionArrowLength(speed), x = (52 - length) / 2, tip = x + length;
      return `<span><svg width="52" height="16" aria-hidden="true"><path d="M${x},8 H${tip} M${tip-4},5 L${tip},8 L${tip-4},11"/></svg>${speed === 30 ? "30+" : speed}</span>`;
    }).join("");
    legend.innerHTML = `<strong>Plate motion · cm/year</strong><div class="motion-samples">${samples}</div><span>Slow motion enlarged · logarithmic arrow lengths · capped at 30 cm/year. Hover for exact speed.</span>`;
    $("motion-fullscreen-legend").innerHTML = legend.innerHTML;
  }

  function motionAt(data, plates, u, v) {
    if (v < 0 || v > 1) return null;
    u = ((u % 1) + 1) % 1;
    const cell = Math.min(data.height - 1, Math.floor(v * data.height)) * data.width + Math.min(data.width - 1, Math.floor(u * data.width));
    const omega = plates.get(data.plate[cell]);
    if (!omega || omega.length !== 3 || !omega.every(Number.isFinite)) return null;
    const lon = u * 2 * Math.PI - Math.PI, lat = Math.PI / 2 - v * Math.PI;
    const c = Math.cos(lat), s = Math.sin(lat), cl = Math.cos(lon), sl = Math.sin(lon);
    const r = [c * cl, c * sl, s];
    const velocity = [omega[1] * r[2] - omega[2] * r[1], omega[2] * r[0] - omega[0] * r[2], omega[0] * r[1] - omega[1] * r[0]];
    return { velocity, speed: Math.hypot(...velocity) * 6371 * .1,
      east: -velocity[0] * sl + velocity[1] * cl,
      north: -velocity[0] * s * cl - velocity[1] * s * sl + velocity[2] * c, cosLat: c };
  }

  function drawMotion(data, globe = false) {
    const target = globe ? globeCanvas : canvas, ctx = target.getContext("2d"), rect = target.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return;
    const plates = new Map((data.plates || []).map(p => [p.id, p.angular_velocity]));
    const camera = globe ? DeepTimeGlobe.basis(globeView.pose.lon, globeView.pose.lat) : null;
    const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
    ctx.save();
    // Screen-space sizes and density stay readable at every zoom/DPR. Project
    // globe tangents directly, rather than stretching arrows from its texture.
    ctx.setTransform(target.width / rect.width, 0, 0, target.height / rect.height, 0, 0);
    if (globe) {
      const radius = .44 * Math.min(target.width, target.height) * globeView.pose.zoom;
      ctx.beginPath();
      ctx.ellipse(rect.width / 2, rect.height / 2, radius * rect.width / target.width, radius * rect.height / target.height, 0, 0, 2 * Math.PI);
      ctx.clip();
    }
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    const cols = Math.max(1, Math.floor(rect.width / 60)), rows = Math.max(1, Math.floor(rect.height / 60));
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      const x = (col + .5) * rect.width / cols, y = (row + .5) * rect.height / rows;
      const point = globe ? globeView.renderer.pick(rect.left + x, rect.top + y, globeView.pose) :
        { u: (x / rect.width - .5) / mapView.zoom + mapView.cx, v: (y / rect.height - .5) / mapView.zoom + mapView.cy };
      if (!point) continue;
      const motion = motionAt(data, plates, point.u, point.v);
      if (!motion || motion.speed < 1e-12) continue;
      let dx = globe ? dot(motion.velocity, camera.east) : motion.east / Math.max(1e-8, motion.cosLat) * rect.width / (2 * Math.PI);
      let dy = globe ? -dot(motion.velocity, camera.north) : -motion.north * rect.height / Math.PI;
      const magnitude = Math.hypot(dx, dy);
      if (!(magnitude > 1e-14)) continue;
      const length = motionArrowLength(motion.speed);
      dx *= length / magnitude; dy *= length / magnitude;
      const angle = Math.atan2(dy, dx), tipX = x + dx / 2, tipY = y + dy / 2;
      ctx.beginPath(); ctx.moveTo(x - dx / 2, y - dy / 2); ctx.lineTo(tipX, tipY);
      ctx.moveTo(tipX - Math.cos(angle - .6) * 5, tipY - Math.sin(angle - .6) * 5);
      ctx.lineTo(tipX, tipY); ctx.lineTo(tipX - Math.cos(angle + .6) * 5, tipY - Math.sin(angle + .6) * 5);
      ctx.strokeStyle = "rgba(7,23,33,.85)"; ctx.lineWidth = 3.8; ctx.stroke();
      ctx.strokeStyle = "#ecf4ed"; ctx.lineWidth = 1.5; ctx.stroke();
    }
    ctx.restore();
  }

  function drawTrace() {
    const points = state.inspection.points.filter((p) => p.index <= state.currentIndex && Number.isFinite(p.lon) && Number.isFinite(p.lat) && Number.isFinite(p.elevation_m));
    const current = selectedHistoryPoint(), sx = canvas.width / 360, sy = canvas.height / 180, ratio = canvas.width / 1000;
    context.save(); context.strokeStyle = "#f9eac0"; context.lineWidth = 2 * ratio; context.shadowColor = "#071721"; context.shadowBlur = 3 * ratio;
    if (state.inspection.mode === "material") {
      for (let i = 1; i < points.length; i++) {
        if (points[i].index !== points[i - 1].index + 1) continue;
        const a = points[i - 1], b = points[i]; let dx = b.lon - a.lon;
        if (dx > 180) dx -= 360; else if (dx < -180) dx += 360;
        for (const wrap of [-360, 0, 360]) { context.beginPath(); context.moveTo((a.lon + 180 + wrap) * sx, (90 - a.lat) * sy); context.lineTo((a.lon + dx + 180 + wrap) * sx, (90 - b.lat) * sy); context.stroke(); }
      }
    }
    if (current && Number.isFinite(current.lon) && Number.isFinite(current.lat)) {
      for (const wrap of [-360, 0, 360]) {
        const x = (current.lon + 180 + wrap) * sx, y = (90 - current.lat) * sy;
        context.fillStyle = "#13242a"; context.beginPath(); context.arc(x, y, 6 * ratio, 0, Math.PI * 2); context.fill(); context.stroke();
        context.fillStyle = "#f9eac0"; context.beginPath(); context.arc(x, y, 2 * ratio, 0, Math.PI * 2); context.fill();
      }
    }
    context.restore();
  }

  function updateExportButtons() {
    const disabled = state.mode !== "history" || !state.frame || state.frame.index !== state.currentIndex || state.loadingRun || globeOrientation.pending;
    for (const key of ["download-heightmap", "download-export", "download-preview"]) $(key).disabled = disabled;
    $("download-heightmap").firstChild.nodeValue = "↓ Recorded grid heightmap ";
    if (state.frame) $("download-heightmap").title = `${state.frame.width} × ${state.frame.height}, exactly as recorded. Decode metres = 16-bit pixel value minus 12,000. Use Develop the terrain above for a detailed 8K heightmap.`;
    updateTerrainControls();
    updateGosplControls();
  }

  async function regenerate(showToast = true) {
    if (isRunning() || state.loadingRun) return;
    let config;
    try { config = readConfig(); } catch (error) { toast(error.message, true); return; }
    const ticket = ++state.initialRequest;
    state.initialLoading = true;
    setRunningControls();
    try {
      const world = await api("/api/initial", { config, preset: $("initial-preset").value });
      if (ticket !== state.initialRequest) return;
      if (showToast) rememberEdit(copyWorld(state.initial));
      state.initial = copyWorld({ ...world, world_design: world.world_design ?? config.world_design }); state.hover = null; mapView.rasterData = null;
      updateResolutionNotes();
      setMode("initial"); updateEvents();
      if (showToast) { const preset = $("initial-preset").value; const label = preset === "highland65" ? "Highland 65" : preset === "land65" ? "Land-rich stress" : "Classic Pangaea"; toast(`${label} preset applied. Paint to customize it, or Undo to restore your previous map.`); }
    } catch (error) { toast(error.message, true); }
    finally { if (ticket === state.initialRequest) { state.initialLoading = false; setRunningControls(); } }
  }

  function updateTimeline() {
    const count = state.frames.length;
    $("timeline").max = Math.max(0, count - 1);
    $("timeline").value = Math.min(state.currentIndex, Math.max(0, count - 1));
    $("timeline").disabled = state.loadingRun || !count;
    $("play-button").disabled = state.loadingRun || count < 2;
    $("previous-frame").disabled = state.loadingRun || !count || state.currentIndex <= 0;
    $("next-frame").disabled = state.loadingRun || !count || state.currentIndex >= count - 1;
    $("follow-label").hidden = !isRunning();
    $("jump-button").disabled = state.loadingRun || !count;
    $("jump-time").disabled = state.loadingRun || !count;
    $("jump-time").max = Number(state.frames.at(-1)?.time_myr || 0);
    if (document.activeElement !== $("jump-time")) $("jump-time").value = Number(state.frames[state.currentIndex]?.time_myr || 0);
    const duration = state.status.duration_myr || state.status.config?.duration_myr || Number($("duration_myr").value);
    $("timeline-end").textContent = `${Number(duration).toLocaleString()} Myr`;
    $("snapshot-label").textContent = count ? `Snapshot ${state.currentIndex + 1} of ${count}${isRunning() ? " · recording" : ""}` : "Run an experiment to record its history";
    updateExportButtons();
  }

  async function showFrame(index, options = {}) {
    if (!state.frames.length) return;
    index = Math.max(0, Math.min(state.frames.length - 1, Number(index)));
    if (!Number.isInteger(index)) return;
    if (!options.automatic) { state.followLatest = false; $("follow-live").checked = false; }
    if (index !== state.currentIndex && state.inspectionPending) { state.inspectionRequest++; state.inspectionPending = false; updateInspection(); }
    const ticket = ++state.frameRequest;
    const run = state.runId, pose = outputOrientation(), cacheKey = frameCacheKey(index, pose, run);
    state.currentIndex = index;
    updateTimeline();
    updateExportButtons();
    try {
      let frame = state.cache.get(cacheKey);
      if (!frame) {
        frame = await api(orientedPath(`/api/frame?index=${index}`, pose));
        if (ticket !== state.frameRequest || run !== state.runId || !sameOrientation(pose, outputOrientation())) return;
        for (const key of ["elevation", "plate", "crust", "age", "boundary", "domain", "trench", "crustal_thickness_km", "crustal_root_km", "rift_thermal_support_m", "rift_cooling_age_myr", "foreland_deflection_m", "erosion_rate_m_myr", "rift_damage", "rift_strength_relative", "deformation_weight", "geometric_strain_percent", "refinement_level"]) if (Array.isArray(frame[key]?.[0])) frame[key] = frame[key].flat();
        state.cache.set(cacheKey, frame);
        const limit = Math.max(2, Math.min(12, Math.floor(4 * 1024 * 1024 / (frame.width * frame.height))));
        while (state.cache.size > limit) state.cache.delete(state.cache.keys().next().value);
      }
      if (ticket !== state.frameRequest || run !== state.runId || !sameOrientation(pose, outputOrientation())) return;
      state.frame = frame;
      if (options.elevation || state.mode === "initial") state.layer = "elevation";
      setMode("history");
      $("time-value").textContent = Number(frame.time_myr).toLocaleString(undefined, { maximumFractionDigits: 1 });
      updateEvents(); updateTimeline();
    } catch (error) { if (ticket === state.frameRequest) { stopPlayback(); toast(error.message, true); } }
  }

  function setRunningControls() {
    const running = isRunning();
    const busy = isBusy();
    document.body.classList.toggle("is-running", running);
    for (const element of $("config-form").querySelectorAll("input, select")) element.disabled = busy;
    $("remove-ocean-attachments").disabled = busy;
    $("run-button").disabled = busy;
    $("run-button").innerHTML = running ? '<span class="loading-ring" style="width:12px;height:12px;border-color:#756944;border-top-color:#d7bd8f"></span> Simulating…' : busy ? 'Preparing world…' : state.status.state === "paused" ? '<span aria-hidden="true">▶</span> Run new experiment' : '<span aria-hidden="true">▶</span> Run simulation';
    for (const key of ["regenerate", "clear-world", "import-initial", "edit-world", "initial-preset"]) $(key).disabled = busy;
    $("saved-runs").disabled = busy;
    $("load-run").disabled = busy || !$("saved-runs").value;
    for (const button of document.querySelectorAll(".brush-button")) button.disabled = busy || state.mode !== "initial";
    $("brush-size").disabled = busy || state.mode !== "initial";
    const status = state.status, paused = status.state === "paused", transition = state.runTransition || ["pausing", "resuming"].includes(status.state);
    let errorNote = $("simulation-error");
    if (!errorNote) {
      errorNote = document.createElement("p");
      errorNote.id = "simulation-error";
      errorNote.setAttribute("role", "alert");
      errorNote.style.cssText = "color:#ef837b;padding:12px 16px;white-space:normal;overflow-wrap:anywhere";
      $("map-shell").insertAdjacentElement("afterend", errorNote);
    }
    errorNote.hidden = status.state !== "error";
    errorNote.textContent = status.state === "error" ? `Simulation stopped: ${status.error || "Unknown error. The server log has more information."} Saved history remains available.` : "";
    const resumable = Boolean(status.can_resume), legacyStopped = ["cancelled", "interrupted"].includes(status.state);
    $("run-progress").hidden = !running && !paused && !legacyStopped && !["complete", "error"].includes(status.state);
    $("pause-button").hidden = !running && !paused && !legacyStopped;
    $("pause-button").textContent = status.state === "pausing" ? "Pausing…" : status.state === "resuming" ? "Resuming…" : paused || legacyStopped ? "Resume" : "Pause";
    $("pause-button").disabled = transition || !(status.state === "running" || resumable);
    $("resume-note").hidden = !paused && !legacyStopped && !transition;
    $("resume-note").textContent = status.state === "pausing" ? "Finishing the current step and saving an exact checkpoint…" : status.state === "resuming" ? "Restoring this experiment's checkpoint…" : resumable ? "Resume continues this same experiment from its saved checkpoint. Starting a new experiment keeps this paused record available." : status.resume_reason || "This older stopped run has no saved checkpoint. Its history remains reviewable; it cannot resume.";
    updateEditorControls();
    $("map-shell").classList.toggle("editing", !busy && state.mode === "initial");
    updateOrientationControls();
    updateGosplControls();
    updateTimeline();
    if (state.mode === "history") {
      $("mode-badge").textContent = paused ? "PAUSED EXPERIMENT" : status.state === "pausing" ? "SAVING CHECKPOINT" : status.state === "resuming" ? "RESUMING" : running ? "SIMULATING" : "RECORDED HISTORY";
      $("mode-badge").className = `mode-badge ${running ? "running" : "history"}`;
    }
  }

  async function applyStatus(status) {
    const previousState = state.status.state;
    const previousCount = state.frames.length;
    const following = state.followLatest && !state.playing && (state.mode === "initial" || state.currentIndex >= previousCount - 1);
    if (state.runId !== status.run_id) {
      state.cache.clear(); state.frameRequest++; state.currentIndex = 0; state.frame = null;
      state.runId = status.run_id;
      syncOrientationContext();
      reconcileTerrainExperiment();
      resetHistory();
    }
    state.status = status;
    let transitionNote = $("source-transition-note");
    if (!transitionNote) {
      transitionNote = document.createElement("p");
      transitionNote.id = "source-transition-note";
      transitionNote.className = "hint";
      transitionNote.setAttribute("role", "note");
      $("progress-detail").insertAdjacentElement("afterend", transitionNote);
    }
    const sourceBoundary = status.source_transition?.parent_epoch_myr;
    transitionNote.hidden = !Number.isFinite(sourceBoundary);
    transitionNote.textContent = Number.isFinite(sourceBoundary)
      ? `Model continued after a numerical repair at ${sourceBoundary} Myr. Earlier history remains in the original saved experiment.` : "";
    state.frames = status.frames || [];
    captureTiming("simulation", status, status.run_id);
    if (state.recordCount !== state.frames.length) refreshRecord();
    setRunningControls(); updateTimeline();
    const progress = Number(status.progress || 0);
    $("progress-bar").value = progress > 1 ? progress / 100 : progress;
    $("progress-label").textContent = `${Math.round(Number(status.time_myr || 0)).toLocaleString()} / ${Number(status.duration_myr || status.config?.duration_myr || $("duration_myr").value).toLocaleString()} Myr`;
    $("progress-detail").textContent = `${state.frames.length} snapshots saved · ${Math.round((progress > 1 ? progress / 100 : progress) * 100)}% complete`;
    $("connection-status").textContent = `LOCAL ENGINE · ${status.state === "running" ? "RUNNING" : "CONNECTED"}`;
    if (state.frames.length && !globeOrientation.pending && (!state.frame || (following && previousCount !== state.frames.length))) await showFrame(state.frames.length - 1, { automatic: true });
    if (state.inspection && previousCount !== state.frames.length) updateInspection();
    if (state.status !== status) return;
    if (["running", "pausing", "resuming"].includes(previousState) && ["complete", "paused", "cancelled", "interrupted", "error"].includes(status.state)) {
      if (status.state === "complete") toast(`${Number(status.time_myr).toLocaleString()} million years recorded. Explore the history or export your world.`);
      else if (status.state === "paused") toast("Experiment paused and checkpoint saved. Resume continues this same history.");
      else if (status.state === "error") toast(status.error || "The simulation stopped because of an error. Saved snapshots remain reviewable.", true);
      else toast("Simulation stopped. Its saved history is ready to review.");
      await refreshRuns();
    }
  }

  async function poll() {
    if (state.pollBusy || state.starting || state.loadingRun) return;
    state.pollBusy = true;
    const epoch = state.pollEpoch;
    try { const status = await api("/api/status"); if (epoch === state.pollEpoch) await applyStatus(status); state.hasConnected = true; }
    catch (error) { $("connection-status").textContent = "LOCAL ENGINE · CONNECTION LOST"; }
    finally { state.pollBusy = false; }
  }

  async function pauseOrResume() {
    if (state.runTransition || state.loadingRun) return;
    const resume = state.status.state !== "running";
    if (resume && !state.status.can_resume) return;
    const previous = state.status, run = state.runId;
    state.runTransition = true; state.pollEpoch++;
    state.status = { ...previous, state: resume ? "resuming" : "pausing" };
    if (resume) captureTiming("simulation", { ...previous, timing_state: "estimating", state: "resuming" }, run);
    setRunningControls();
    try {
      const status = await api(resume ? "/api/resume" : "/api/pause", { run_id: run });
      if (run !== state.runId) return;
      if (resume && state.mode === "initial") setMode("history");
      await applyStatus(status);
    } catch (error) { if (run === state.runId) { state.status = previous; captureTiming("simulation", previous, run); } toast(error.message, true); }
    finally { state.runTransition = false; setRunningControls(); await poll(); }
  }

  // Rift-inheritance inputs sit in a collapsed <details>, and in Legacy inside
  // hidden controls. When the law is off they are not sent, so an invalid leftover
  // returns to its page default instead of blocking the start; when it is on, the
  // panel opens so the browser can point at the invalid value.
  function prepareRiftInputsForValidation() {
    const details = $("enhanced-rifting-seed").closest?.("details");
    if (!details) return;
    const inputs = [...details.querySelectorAll("input")];
    const active = $("physics_profile").value === "reviewed_v1" && $("enhanced-rifting-enabled").checked;
    for (const input of inputs) if (!active && !input.checkValidity()) input.value = input.defaultValue;
    if (active && inputs.some(input => !input.checkValidity())) details.open = true;
  }

  async function runSimulation(event) {
    event.preventDefault();
    prepareRiftInputsForValidation();
    if (isBusy() || !$("config-form").reportValidity()) return;
    stopPlayback();
    let config;
    try { config = readConfig(); } catch (error) { toast(error.message, true); return; }
    if (!state.initial) await regenerate(false);
    if (!state.initial) return;
    if (state.initial.width !== config.width || state.initial.height !== config.height) { rememberEdit(copyWorld(state.initial)); state.initial = regridWorld(state.initial, config.width, config.height); }
    state.starting = true;
    captureTiming("simulation", { elapsed_seconds: 0, eta_seconds: null, timing_state: "estimating", state: "running" }, "pending-simulation");
    state.pollEpoch++;
    setRunningControls();
    try {
      const result = await api("/api/run", { config, initial: state.initial });
      state.frameRequest++; state.cache.clear(); state.frame = null; state.frames = []; state.currentIndex = 0; state.runId = result.run_id;
      $("saved-runs").value = "";
      reconcileTerrainExperiment();
      resetHistory();
      state.status = { state: "running", run_id: result.run_id, duration_myr: config.duration_myr };
      state.starting = false;
      setRunningControls(); updateTimeline();
      toast("Simulation started. New snapshots will appear as the world evolves.");
      await poll();
      await refreshRuns();
    } catch (error) { state.starting = false; captureTiming("simulation", state.status, state.runId); setRunningControls(); toast(error.message, true); }
  }

  async function refreshRuns() {
    try {
      const result = await api("/api/runs");
      const select = $("saved-runs"), previous = select.value || state.runId;
      select.replaceChildren(new Option(result.runs?.length ? "Choose an experiment…" : "No saved experiments yet", ""));
      for (const run of result.runs || []) {
        const date = new Date(run.created);
        const label = run.title || (Number.isNaN(date.getTime()) ? run.run_id : date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }));
        select.add(new Option(`${label} · ${Number(run.time_myr || 0).toLocaleString()} Myr · ${run.state}`, run.run_id));
      }
      if (Array.from(select.options).some((option) => option.value === previous)) select.value = previous;
      $("load-run").disabled = isBusy() || !select.value;
    } catch (error) { $("connection-status").textContent = "LOCAL ENGINE · CONNECTION LOST"; }
  }

  async function loadRun() {
    if (!$("saved-runs").value || isBusy()) return;
    stopPlayback();
    state.loadingRun = true;
    state.frameRequest++;
    state.inspectionRequest++; state.inspectionPending = false;
    setRunningControls();
    state.pollEpoch++;
    try {
      const status = await api("/api/load", { run_id: $("saved-runs").value });
      fillConfig(status.config);
      state.initial = copyWorld(await api("/api/initial")); resetEditHistory();
      state.cache.clear(); state.frameRequest++; state.frame = null; state.frames = [];
      await applyStatus(status);
      toast(`Opened ${status.state === "complete" ? "completed" : "saved"} experiment with ${state.frames.length} snapshots.`);
    } catch (error) { toast(error.message, true); }
    finally { state.loadingRun = false; setRunningControls(); }
  }

  function stopPlayback() {
    state.playing = false; clearTimeout(state.playTimer); clearTimeout(state.scrubTimer);
    $("play-button").textContent = "▶"; $("play-button").setAttribute("aria-label", "Play history");
    updateGlobePlayback();
  }

  async function playNext() {
    if (!state.playing) return;
    if (state.currentIndex >= state.frames.length - 1) { stopPlayback(); return; }
    await showFrame(state.currentIndex + 1);
    if (state.playing) state.playTimer = setTimeout(playNext, Number($("play-speed").value));
  }

  async function togglePlayback() {
    if (state.playing) { stopPlayback(); return; }
    if (state.frames.length < 2) return;
    if (state.mode === "initial" || state.currentIndex >= state.frames.length - 1) await showFrame(0);
    state.playing = true; $("play-button").textContent = "Ⅱ"; $("play-button").setAttribute("aria-label", "Pause history");
    updateGlobePlayback();
    state.playTimer = setTimeout(playNext, Number($("play-speed").value));
  }

  function mapPoint(event) {
    const data = state.mode === "initial" ? state.initial : state.frame;
    if (!data) return null;
    const rect = canvas.getBoundingClientRect();
    const fx = (event.clientX - rect.left) / rect.width, fy = (event.clientY - rect.top) / rect.height;
    // With wrapped panning the view can extend past the antimeridian, so fold x back onto
    // [0, 1) instead of clamping it to the edge cell.
    const x = mapView.cx + (fx - .5) / mapView.zoom, xw = x - Math.floor(x);
    const y = mapView.cy + (fy - .5) / mapView.zoom;
    // u,v are the exact fractional position on the map. Inspection and painting
    // want the integer cell, but the distance measure must not be quantised to it:
    // a cell is 1.875 degrees (~208 km at the equator) here, so two clicks inside
    // one cell produced an identical reading and the tool looked stuck.
    return { x: Math.min(data.width - 1, Math.floor(xw * data.width)),
             y: Math.min(data.height - 1, Math.max(0, Math.floor(y * data.height))),
             u: xw, v: Math.min(1, Math.max(0, y)) };
  }

  function updateCollisionHistory() {
    const frame = state.mode === "history" ? state.frame : null;
    $("collision-history").hidden = frame?.collision_contact_version !== 1;
    if ($("collision-history").hidden) return;
    const rows = frame.collision_contacts || [], diagnostic = frame.collision_diagnostics || {};
    const active = rows.filter(row => row.state === "active").length;
    $("collision-count").textContent = `${active} ACTIVE · ${rows.length} RETAINED`;
    const buried = Number(diagnostic.buried_area_quadrature_km2), stack = Number(diagnostic.maximum_pair_stack_thickness_km);
    $("collision-description").textContent = `${formatTime(frame.time_myr)} Myr · ${active ? "Persistent contacts connect overlapping material sheets." : "No active different-owner contact at this epoch."}${Number.isFinite(buried) ? ` Approximately ${Math.round(buried).toLocaleString()} km² of material is covered, estimated from four samples per contacted triangle.` : ""}${Number.isFinite(stack) && stack > 0 ? ` Maximum recorded pair of crustal columns: ${stack.toFixed(1)} km combined.` : ""}`;
    const container = $("collision-contacts"); container.replaceChildren();
    const sorted = [...rows].sort((a,b) => (a.state === "quiet") - (b.state === "quiet") || b.overlap_area_km2 - a.overlap_area_km2);
    for (const row of sorted.slice(0, 60)) {
      const line = document.createElement("p"); line.className = "hint";
      line.textContent = `Contact ${row.id} · ${row.state} · sheet ${row.top_sheet} above ${row.under_sheet} · began ${formatTime(row.started_myr)} Myr · last contact ${formatTime(row.last_seen_myr)} Myr · ${Math.round(row.overlap_area_km2 || 0).toLocaleString()} km² current overlap · ${(row.cumulative_convergence_km || 0).toFixed(1)} km accumulated convergence · suture ${Math.round((row.suture_strength || 0) * 100)}%.`;
      container.append(line);
    }
    if (rows.length > 60) { const line = document.createElement("p"); line.className = "hint"; line.textContent = `Showing 60 of ${rows.length} contacts; all contacts are retained in the experiment export.`; container.append(line); }
    if (!rows.length) container.textContent = "No material contact recorded yet.";
  }

  function updateAuthoredHistory() {
    const frame = state.mode === "history" ? state.frame : null;
    const design = normalizeWorldDesign(frame?.world_design);
    $("authored-history").hidden = !frame || !design.interventions.length;
    if ($("authored-history").hidden) return;
    const active = design.enabled ? design.interventions.filter(row => row.strength > 0 && row.start_myr <= frame.time_myr && frame.time_myr < row.end_myr) : [];
    $("authored-count").textContent = `${active.length} ACTIVE · REVISION ${design.revision}`;
    $("authored-description").textContent = `${design.enabled ? "Enabled" : "Disabled"} for this experiment. Cyan marks protection; pink marks preferred weak regions. These are authored geographic controls, recorded separately from inferred tectonic events.`;
    const container = $("authored-regions"); container.replaceChildren();
    for (const row of design.interventions) {
      const line = document.createElement("p"); line.className = "hint";
      line.textContent = `${row.id} · ${row.kind} · ${active.includes(row) ? "active now" : "inactive now"} · ${row.start_myr}–${row.end_myr} Myr · strength ${row.strength}: ${row.reason}`;
      container.append(line);
    }
  }

  function normalizeWorldDesign(value = null) {
    value ??= {};
    if (typeof value !== "object" || Array.isArray(value) || Object.keys(value).some(key => !["version", "enabled", "revision", "interventions"].includes(key))) throw new Error("Invalid world design object.");
    const version = value.version ?? 1, enabled = value.enabled ?? false, revision = value.revision ?? 0, rows = value.interventions ?? [];
    if (version !== 1 || typeof enabled !== "boolean" || !Number.isInteger(revision) || revision < 0 || revision > 2147483647 || !Array.isArray(rows) || rows.length > 32) throw new Error("World design needs version 1, a valid revision and at most 32 regions.");
    const ids = new Set(), limits = { lon_deg: [-180, 180], lat_deg: [-90, 90], radius_deg: [.25, 45], strength: [0, 1], start_myr: [0, 1000], end_myr: [0, 1000] };
    return { version, enabled, revision, interventions: rows.map(row => {
      if (!row || typeof row !== "object" || Object.keys(row).sort().join(",") !== ["id", "kind", "reason", ...Object.keys(limits)].sort().join(",") || typeof row.id !== "string" || !/^[A-Za-z0-9_-]{1,64}$/.test(row.id) || ids.has(row.id) || !["protected", "weak"].includes(row.kind) || typeof row.reason !== "string" || !row.reason.trim() || row.reason.trim().length > 500) throw new Error("Each authored region needs a unique ID, effect and reason.");
      ids.add(row.id);
      const result = { id: row.id, kind: row.kind, reason: row.reason.trim() };
      for (const [key, [low, high]] of Object.entries(limits)) {
        if (typeof row[key] !== "number" || !Number.isFinite(row[key]) || row[key] < low || row[key] > high) throw new Error(`${key.replaceAll("_", " ")} must be between ${low} and ${high}.`);
        result[key] = row[key];
      }
      if (result.end_myr <= result.start_myr) throw new Error("The region must end after its starting time.");
      result.lon_deg = ((result.lon_deg + 180) % 360 + 360) % 360 - 180;
      return result;
    }) };
  }

  function readDesignRegion(preview = false) {
    const row = { id: "preview", kind: $("design-kind").value, reason: $("design-reason").value.trim() || (preview ? "Region preview" : "") };
    for (const [key, id] of [["lon_deg", "lon"], ["lat_deg", "lat"], ["radius_deg", "radius"], ["strength", "strength"], ["start_myr", "start"], ["end_myr", "end"]]) {
      if (!$("design-" + id).value.trim()) throw new Error("Complete the region coordinates, strength and interval.");
      row[key] = Number($("design-" + id).value);
    }
    return normalizeWorldDesign({ interventions: [row] }).interventions[0];
  }

  function editWorldDesign(change) {
    if (isBusy() || state.mode !== "initial" || !state.initial) return;
    const before = copyWorld(state.initial), design = normalizeWorldDesign(state.initial.world_design);
    change(design); design.revision++;
    state.initial.world_design = normalizeWorldDesign(design);
    rememberEdit(before); designEditor.preview = null; designEditor.picking = false;
    updateEditorControls(); render();
  }

  function addDesignRegion() {
    try {
      const row = readDesignRegion();
      editWorldDesign(design => {
        let serial = design.revision + 1;
        while (design.interventions.some(item => item.id === `region-${serial}`)) serial++;
        row.id = `region-${serial}`; design.interventions.push(row); design.enabled = true;
      });
      $("design-regions").value = row.id; updateWorldDesignControls();
      toast("Region added to the next experiment. Save world preserves it; Undo restores the previous design.");
    } catch (error) { toast(error.message, true); }
  }

  function updateWorldDesignControls() {
    const disabled = isBusy() || state.mode !== "initial" || !state.initial;
    const design = normalizeWorldDesign(state.initial?.world_design);
    for (const id of ["world-design-enabled", "design-kind", "design-lon", "design-lat", "design-radius", "design-strength", "design-start", "design-end", "design-reason", "design-pick", "design-preview", "design-add"]) $(id).disabled = disabled;
    $("design-add").disabled = disabled || design.interventions.length >= 32;
    $("world-design-enabled").checked = design.enabled;
    $("design-pick").textContent = designEditor.picking ? "Click the map…" : "Pick on map";
    $("design-pick").setAttribute("aria-pressed", String(designEditor.picking));
    const select = $("design-regions"), previous = select.value;
    select.replaceChildren(...(design.interventions.length ? design.interventions.map(row => new Option(`${row.id} · ${row.kind} · ${row.start_myr}–${row.end_myr} Myr`, row.id)) : [new Option("No authored regions", "")]));
    select.value = design.interventions.some(row => row.id === previous) ? previous : design.interventions[0]?.id || "";
    select.disabled = disabled || !design.interventions.length;
    $("design-remove").disabled = disabled || !select.value;
    const canBranch = !disabled && state.status.state === "paused" && state.status.can_resume && state.runId === state.status.run_id;
    $("design-branch").disabled = !canBranch;
    $("design-branch").textContent = canBranch ? `Branch at ${formatTime(Number(state.status.time_myr || 0))} Myr` : "Branch paused experiment";
    $("design-branch-note").textContent = canBranch ? `Uses the paused crust and recorded motion, with this regional design. New regions must start at or after ${formatTime(Number(state.status.time_myr || 0))} Myr. The new branch starts paused; its source remains saved.` : "Open a compatible paused checkpoint, then choose Edit to prepare a branch. Older engines retain their own saved continuation.";
    const selected = design.interventions.find(row => row.id === select.value);
    $("design-status").textContent = selected ? `${design.enabled ? "Enabled" : "Disabled"} · revision ${design.revision} · ${design.interventions.length}/32 regions. ${selected.reason} (${selected.lon_deg.toFixed(1)}°, ${selected.lat_deg.toFixed(1)}°; radius ${selected.radius_deg}°; strength ${selected.strength}). Saved with the starting world; Undo restores edits.` : "No authored regions. Save world includes these controls; Undo restores edits.";
  }

  function drawWorldDesign(data, initial) {
    if (!$("design-show").checked) return;
    const design = normalizeWorldDesign(data.world_design);
    const rows = initial ? design.interventions : design.enabled ? design.interventions.filter(row => row.strength > 0 && row.start_myr <= data.time_myr && data.time_myr < row.end_myr) : [];
    context.save();
    for (const row of [...rows, ...(initial && designEditor.preview ? [designEditor.preview] : [])]) {
      const preview = row === designEditor.preview;
      context.fillStyle = preview ? "rgba(255,244,207,.32)" : row.kind === "protected" ? `rgba(110,212,228,${design.enabled ? .22 : .07})` : `rgba(238,150,211,${design.enabled ? .24 : .07})`;
      const point = { x: (row.lon_deg + 180) / 360 * data.width - .5, y: (90 - row.lat_deg) / 180 * data.height - .5 };
      for (const strip of sphericalBrushRows(data, point, row.radius_deg)) for (const [start, stop] of strip.spans) context.fillRect(start / data.width * canvas.width, strip.y / data.height * canvas.height, (stop - start) / data.width * canvas.width, canvas.height / data.height);
    }
    context.restore();
  }

  async function branchWorldDesign() {
    updateWorldDesignControls();
    if ($("design-branch").disabled) return;
    const parent = state.runId, design = normalizeWorldDesign(state.initial?.world_design);
    state.loadingRun = true; state.pollEpoch++; state.frameRequest++;
    state.inspectionRequest++; state.inspectionPending = false; stopPlayback(); setRunningControls();
    try {
      const status = await api("/api/branch", { run_id: parent, world_design: design });
      fillConfig(status.config);
      state.cache.clear(); state.frame = null; state.frames = [];
      await applyStatus(status);
      state.initial = copyWorld(await api("/api/initial")); resetEditHistory();
      $("saved-runs").value = ""; await refreshRuns();
      toast(`Created a separate branch at ${formatTime(Number(status.time_myr || 0))} Myr. Resume continues it with the recorded regional design.`);
    } catch (error) { toast(`Could not branch experiment: ${error.message}`, true); }
    finally { state.loadingRun = false; setRunningControls(); }
  }

  function copyWorld(world) {
    return world ? { ...world,
      ...(world.world_design ? { world_design: normalizeWorldDesign(world.world_design) } : {}),
      ...(world.initial_plate_topology ? { initial_plate_topology: structuredClone(world.initial_plate_topology) } : {}),
      ...(world.initial_subduction ? { initial_subduction: structuredClone(world.initial_subduction) } : {}),
      ...(world.continental_lifecycle ? { continental_lifecycle: structuredClone(world.continental_lifecycle) } : {}),
      ...(world.rift_traction ? { rift_traction: structuredClone(world.rift_traction) } : {}),
      crust: Uint8Array.from(Array.isArray(world.crust?.[0]) ? world.crust.flat() : world.crust) } : null;
  }

  function trimEditHistory() {
    const bytes = () => [...editHistory.undo, ...editHistory.redo].reduce((sum, world) => sum + world.crust.byteLength, 0);
    while (editHistory.undo.length + editHistory.redo.length > editHistory.maxEntries || bytes() > editHistory.maxBytes) {
      if (editHistory.undo.length) editHistory.undo.shift(); else editHistory.redo.shift();
    }
  }

  function rememberEdit(before) {
    if (!before) return;
    editHistory.undo.push(before); editHistory.redo = []; trimEditHistory();
  }

  function resetEditHistory() {
    editHistory.undo = []; editHistory.redo = []; editHistory.before = null; editHistory.changed = 0;
    updateEditorControls();
  }

  function beginEdit() {
    if (!editHistory.before) { editHistory.before = copyWorld(state.initial); editHistory.changed = 0; }
  }

  function finishEdit() {
    if (editHistory.changed) rememberEdit(editHistory.before);
    editHistory.before = null; editHistory.changed = 0;
    updateEditorControls(); updateStats();
  }

  function restoreEdit(redo = false) {
    if (isBusy() || state.mode !== "initial" || state.drawing) return;
    const from = redo ? editHistory.redo : editHistory.undo, to = redo ? editHistory.undo : editHistory.redo;
    if (!from.length) return;
    to.push(copyWorld(state.initial)); state.initial = from.pop(); trimEditHistory();
    fillConfig({ width: state.initial.width }); mapView.rasterData = null; state.hover = null;
    updateEditorControls(); updateStats(); render();
  }

  function initialCoverage(world) {
    if (!world) return { continental: 0, craton: 0 };
    let total = 0, continental = 0, craton = 0;
    for (let y = 0; y < world.height; y++) {
      const weight = Math.sin(Math.PI / 2 - y * Math.PI / world.height) - Math.sin(Math.PI / 2 - (y + 1) * Math.PI / world.height);
      total += weight * world.width;
      for (let x = 0; x < world.width; x++) {
        const kind = world.crust[y * world.width + x];
        if (kind > 0) continental += weight;
        if (kind === 2) craton += weight;
      }
    }
    return { continental: continental / total, craton: craton / total };
  }

  function updateEditorControls() {
    const disabled = isBusy() || state.mode !== "initial";
    updateWorldDesignControls();
    for (const id of ["paint-tool", "fill-tool", "brush-size"]) $(id).disabled = disabled;
    $("brush-size").disabled = disabled || state.editorTool !== "brush";
    $("undo-edit").disabled = disabled || state.drawing || !editHistory.undo.length;
    $("redo-edit").disabled = disabled || state.drawing || !editHistory.redo.length;
    for (const [id, tool] of [["paint-tool", "brush"], ["fill-tool", "fill"]]) {
      $(id).classList.toggle("selected", state.editorTool === tool); $(id).setAttribute("aria-pressed", String(state.editorTool === tool));
    }
    $("brush-note").textContent = state.editorTool === "fill" ? "Click to replace a connected region of one material. Fill wraps around the globe; Undo restores it." : "Spherical radius stays the same at every resolution, including across the poles. A tiny brush still paints one cell.";
    if (state.initial) {
      const coverage = initialCoverage(state.initial);
      $("editor-coverage").textContent = `${(coverage.continental * 100).toFixed(1)}% continental crust · ${(coverage.craton * 100).toFixed(1)}% craton (part of that crust) · globe area`;
    }
  }

  function regridWorld(world, width, height) {
    const crust = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      const sourceY = Math.min(world.height - 1, Math.floor((y + .5) * world.height / height));
      for (let x = 0; x < width; x++) {
        const sourceX = Math.floor((x + .5) * world.width / width) % world.width;
        crust[y * width + x] = world.crust[sourceY * world.width + sourceX];
      }
    }
    return { ...world, width, height, crust };
  }

  function changeResolution() {
    if (isBusy()) return;
    updateResolutionNotes();
    const width = Number($("width").value), height = Math.round(width / 2);
    if (!state.initial) { regenerate(false); return; }
    if (state.initial.width === width && state.initial.height === height) return;
    rememberEdit(copyWorld(state.initial)); state.initial = regridWorld(state.initial, width, height);
    state.hover = null; mapView.rasterData = null; state.followLatest = false;
    setMode("initial");
    toast(`Painted world retained at ${width.toLocaleString()} × ${height.toLocaleString()}. Coarser grids can lose small features; Undo restores the previous grid.`);
  }

  function sphericalBrushRows(world, point, radiusDegrees) {
    const w = world.width, h = world.height, radius = Math.max(.25, Math.min(45, radiusDegrees)) * Math.PI / 180;
    const latitude = Math.PI / 2 - (point.y + .5) / h * Math.PI;
    const longitude = (point.x + .5) / w * 2 * Math.PI - Math.PI;
    const first = Math.max(0, Math.ceil((Math.PI / 2 - latitude - radius) * h / Math.PI - .5 - 1e-10));
    const last = Math.min(h - 1, Math.floor((Math.PI / 2 - latitude + radius) * h / Math.PI - .5 + 1e-10));
    const rows = [], cosine = Math.cos(radius), sinCenter = Math.sin(latitude), cosCenter = Math.cos(latitude);
    for (let y = first; y <= last; y++) {
      const lat = Math.PI / 2 - (y + .5) / h * Math.PI;
      const quotient = (cosine - sinCenter * Math.sin(lat)) / (cosCenter * Math.cos(lat));
      if (quotient > 1 + 1e-12) continue;
      const half = quotient <= -1 ? Math.PI : Math.acos(Math.max(-1, Math.min(1, quotient)));
      const centerX = (longitude + Math.PI) / (2 * Math.PI) * w - .5;
      const start = Math.ceil(centerX - half * w / (2 * Math.PI) - 1e-10), stop = Math.floor(centerX + half * w / (2 * Math.PI) + 1e-10) + 1;
      const count = Math.min(w, stop - start);
      if (count <= 0) continue;
      const x = ((start % w) + w) % w;
      const spans = count === w ? [[0, w]] : x + count <= w ? [[x, x + count]] : [[x, w], [0, x + count - w]];
      rows.push({ y, spans });
    }
    if (!rows.length) rows.push({ y: Math.max(0, Math.min(h - 1, Math.round(point.y))), spans: [[((Math.round(point.x) % w) + w) % w, ((Math.round(point.x) % w) + w) % w + 1]] });
    return rows;
  }

  function stamp(point) {
    mapView.rasterData = null;
    const world = state.initial;
    for (const row of sphericalBrushRows(world, point, state.brushSize)) {
      for (const [start, stop] of row.spans) {
        for (let x = start; x < stop; x++) {
          const cell = row.y * world.width + x;
          if (world.crust[cell] !== state.brush) { world.crust[cell] = state.brush; editHistory.changed++; }
        }
      }
    }
  }

  function fillMaterial(world, startCell, replacement) {
    const original = world.crust[startCell];
    if (original === replacement) return 0;
    const w = world.width, h = world.height, queue = new Int32Array(w * h);
    let head = 0, tail = 0;
    const add = (cell) => { if (world.crust[cell] === original) { world.crust[cell] = replacement; queue[tail++] = cell; } };
    add(startCell);
    while (head < tail) {
      const cell = queue[head++], y = Math.floor(cell / w), x = cell % w;
      add(y * w + (x + w - 1) % w); add(y * w + (x + 1) % w);
      if (y > 0) add(cell - w); else { add((x + Math.floor(w / 2)) % w); if (w % 2) add((x + Math.ceil(w / 2)) % w); }
      if (y < h - 1) add(cell + w); else { add(y * w + (x + Math.floor(w / 2)) % w); if (w % 2) add(y * w + (x + Math.ceil(w / 2)) % w); }
    }
    return tail;
  }

  function paintTo(point) {
    if (!point || !state.initial || state.mode !== "initial" || isBusy()) return;
    if (state.lastPoint) {
      let dx = point.x - state.lastPoint.x;
      const dy = point.y - state.lastPoint.y, w = state.initial.width, h = state.initial.height;
      if (dx > w / 2) dx -= w; else if (dx < -w / 2) dx += w;
      const latitude = Math.PI / 2 - (point.y + .5) / h * Math.PI;
      const angularDistance = Math.hypot(dx * 360 / w * Math.cos(latitude), dy * 180 / h);
      const steps = Math.max(1, Math.ceil(angularDistance / Math.max(.25, state.brushSize / 2)));
      for (let j = 1; j <= steps; j++) stamp({ x: (state.lastPoint.x + dx * j / steps + w) % w, y: state.lastPoint.y + dy * j / steps });
    } else stamp(point);
    state.lastPoint = point; render();
  }

  function showHover(point) {
    const data = state.mode === "initial" ? state.initial : state.frame;
    if (!point || !data) return;
    const i = point.y * data.width + point.x;
    const longitude = (point.x + .5) / data.width * 360 - 180, latitude = 90 - (point.y + .5) / data.height * 180;
    const coordinates = `${Math.abs(latitude).toFixed(1)}° ${latitude >= 0 ? "N" : "S"} · ${Math.abs(longitude).toFixed(1)}° ${longitude >= 0 ? "E" : "W"}`;
    const values = [coordinates, crustNames[data.crust[i]] || "Crust"];
    if (state.mode === "history") {
      const plate = plateAt(data, i), domain = domainAt(data, i);
      const plateName = plate?.name || `Plate ${data.plate[i]}`;
      values.push(`${Math.round(data.elevation[i]).toLocaleString()} m`, domain?.name || plateName);
      if (domain) values.push(`Motion: ${plateName}`);
      const omega = plate?.angular_velocity;
      if (omega?.length === 3 && Array.from(omega).every(Number.isFinite)) {
        const lon = longitude * Math.PI / 180, lat = latitude * Math.PI / 180;
        const r = [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
        // Saved Euler motion is rad/Myr; R * |omega x r| is km/Myr, or 0.1 cm/year.
        const speed = Math.hypot(omega[1] * r[2] - omega[2] * r[1],
          omega[2] * r[0] - omega[0] * r[2], omega[0] * r[1] - omega[1] * r[0]) * 6371 * .1;
        values.push(`Plate speed here: ${speed > 0 && speed < .001 ? "<0.001" : speed.toFixed(3)} cm/year`);
      } else values.push("Plate speed: not recorded");
      if (data.crust[i] === 0) values.push(`${Number(data.age[i]).toFixed(1)} Myr old`);
      if (data.boundary[i]) values.push(boundaryNames[data.boundary[i]]);
      if (foundationLayers[state.layer]) {
        const value = data[foundationLayers[state.layer].field]?.[i];
        values.push(foundationLayers[state.layer].materialOnly && data.crust[i] === 0 ? "Material deformation: unavailable over ocean crust"
          : state.layer === "deformation" ? `Belt participation: ${Number.isFinite(value) ? value === 0 ? "Rigid" : `${Math.round(value * 100)}%` : "Not recorded"}`
          : state.layer === "strain" ? `Area change this step: ${Number.isFinite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(3)}%${value < 0 ? " · shortening" : value > 0 ? " · stretching" : ""}` : "Not recorded"}`
          : state.layer === "refinement" ? `Mesh detail: ${Number.isFinite(value) && value >= 0 ? `Level ${Math.round(value)}` : "Not recorded"}`
          : state.layer === "damage" ? `Rift damage: ${Number.isFinite(value) && value >= 0 && value <= 1 ? `${Math.round(value * 100)}%` : "Not recorded"}`
          : state.layer === "weakness" ? `Relative lithospheric strength: ${Number.isFinite(value) && value > 0 ? value.toFixed(2) : "Not recorded"} · lower is weaker`
          : state.layer === "thickness" ? `Crust thickness: ${Number.isFinite(value) && value >= 0 ? `${value.toFixed(1)} km` : "Not recorded"}`
          : `Foreland deflection: ${Number.isFinite(value) && value >= 0 ? formatMetres(value) : "Not recorded"}`);
      }
      if (hasGridField(data, "trench") && data.trench[i] > 0) {
        const trench = data.trench_systems?.find((row) => row.id === data.trench[i]);
        if (trench) values.push(`Trench ${trench.id}${trench.phase ? ` · ${String(trench.phase).replaceAll("_", " ")}` : ""}`);
      }
      if (state.layer === "loading") values.push(...riftLoadingHover(plate));
    }
    $("map-hover").classList.toggle("loading-hover", state.mode === "history" && state.layer === "loading");
    $("map-hover").textContent = state.mode === "history" && state.layer === "loading" ? values.join("\n") : values.join("  /  "); $("map-hover").hidden = false;
  }

  function download(path) {
    const a = document.createElement("a"); a.href = path; a.download = ""; document.body.append(a); a.click(); a.remove();
  }

  function exportInitial() {
    if (!state.initial) return;
    let config;
    try { config = readConfig(); } catch (error) { toast(error.message, true); return; }
    const json = { format: "deep-time-initial-v1", description: "0 oceanic crust, 1 continental crust, 2 stable craton. Row-major, north to south, longitude wraps.", ...state.initial, crust: Array.from(state.initial.crust), config };
    const url = URL.createObjectURL(new Blob([JSON.stringify(json)], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = `deep-time-starting-world-seed-${$("seed").value}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function importInitial(event) {
    const file = event.target.files[0];
    if (!file) return;
    try {
      if (file.size > 30 * 1024 * 1024) throw new Error("This world file is too large. Import a JSON crust map smaller than 30 MB.");
      const payload = JSON.parse(await file.text()), world = payload.initial || payload;
      const w = Number(world.width), h = Number(world.height), crust = Array.isArray(world.crust) ? world.crust.flat() : [];
      if (!Number.isInteger(w) || !Number.isInteger(h) || w < 64 || w > 2048 || h !== Math.round(w / 2)) throw new Error("Use a world 64–2,048 cells wide with height equal to half its width.");
      if (crust.length !== w * h || !crust.every((value) => [0, 1, 2].includes(value))) throw new Error("The crust array must have width × height cells: 0 ocean, 1 continent, or 2 craton.");
      const worldDesign = normalizeWorldDesign(world.world_design ?? payload.config?.world_design);
      const plateTopology = world.initial_plate_topology == null ? undefined : structuredClone(world.initial_plate_topology);
      const initialSubduction = world.initial_subduction == null ? undefined : structuredClone(world.initial_subduction);
      const continentalLifecycle = world.continental_lifecycle == null ? undefined : structuredClone(world.continental_lifecycle);
      const riftTraction = world.rift_traction == null ? undefined : structuredClone(world.rift_traction);
      rememberEdit(copyWorld(state.initial));
      fillConfig({ ...(payload.config || {}), width: w });
      state.initial = { width: w, height: h, crust: Uint8Array.from(crust), world_design: worldDesign,
        ...(plateTopology == null ? {} : { initial_plate_topology: plateTopology }),
        ...(initialSubduction == null ? {} : { initial_subduction: initialSubduction }),
        ...(continentalLifecycle == null ? {} : { continental_lifecycle: continentalLifecycle }),
        ...(riftTraction == null ? {} : { rift_traction: riftTraction }) }; state.hover = null; mapView.rasterData = null;
      updateResolutionNotes();
      setMode("initial"); updateEvents(); toast(`Imported ${w} × ${h} starting world. Your map is ready to simulate.`);
    } catch (error) { toast(`Could not import world: ${error.message}`, true); }
    finally { event.target.value = ""; }
  }

  function updateOceanAttachmentNote() {
    const entries = oceanAttachmentDraft.entries;
    $("ocean-attachment-controls").hidden = !entries.length;
    if (!entries.length) { $("ocean-attachment-summary").textContent = ""; return; }
    const connections = entries.map(entry => `water region ${entry.region_index + 1} → continental plate ${entry.continental_plate_uid}`).join("; ");
    let note = `Saved initial connections (${oceanAttachmentDraft.plateCount} ocean regions): ${connections}. These regions share their continent's plate motion. Region numbers refer to this starting world; review connections after editing it.`;
    if ($("physics_profile").value !== "reviewed_v1") note += " Inactive in Legacy: a new Legacy experiment omits these connections. Switching back to revised physics restores them in this form.";
    else if (!$("primordial-ocean-enabled").checked) note += " To start or export, enable ocean splitting again or remove these connections.";
    else if (Number($("primordial-ocean-count").value) !== oceanAttachmentDraft.plateCount) note += ` To start or export, restore ${oceanAttachmentDraft.plateCount} ocean regions or remove these connections. Region assignments will not be remapped automatically.`;
    $("ocean-attachment-summary").textContent = note;
  }

  function removeOceanAttachments() {
    if (isBusy()) return;
    oceanAttachmentDraft.entries = [];
    oceanAttachmentDraft.plateCount = null;
    updateResolutionNotes();
    toast("Saved ocean–continent connections removed from the next experiment. Ocean regions will start as independent plates.");
  }

  function updateResolutionNotes() {
    const width = Number($("width").value), duration = Number($("duration_myr").value), cadence = Number($("snapshot_myr").value);
    const count = Math.ceil(duration / Math.max(.5, cadence)) + 1;
    const controlCells = 20 * 4 ** Number($("mesh_level").value);
    const controlScale = Math.sqrt(4 * Math.PI * 6371 ** 2 / controlCells);
    $("grid-resolution-note").textContent = `${controlCells.toLocaleString()} tectonic control cells · approximately ${Math.round(controlScale)} km characteristic cell scale. Display pixels: ${(40030 / width).toFixed(1)} km at the equator. Material geometry can be finer; increasing image size does not refine the dynamics. ${count.toLocaleString()} saved times.`;
    const reviewed = $("physics_profile").value === "reviewed_v1";
    const effective = reviewed && $("subduction-force-law").value === "effective";
    const inheritedDeclaration = state.initial?.initial_subduction;
    if (effective && !state.primordialDeclarationEdited && inheritedDeclaration?.enabled) {
      $("primordial-margin-fraction").value = inheritedDeclaration.target_margin_fraction ?? 1;
      $("primordial-selection-seed").value = inheritedDeclaration.selection_seed ?? 0;
    }
    $("effective-subduction-controls").hidden = !effective;
    $("effective-subduction-force").disabled = !effective;
    $("subduction_response").disabled = effective;
    $("retained_phases").disabled = effective;
    $("subduction-response-field").hidden = effective;
    $("retained-phases-field").hidden = effective;
    $("primordial-slab-depth-field").hidden = effective;
    $("primordial-slab-depth").disabled = effective;
    $("primordial-subduction-note").textContent = effective ? "Declare established trench arcs and their incoming plates. No slab inventory is created. A target below 1 selects whole connected arcs, so the realized share can exceed the target. Constant pull persists through quiet motion; buoyant arrival stops the local pull and enters collision." : "Oceanic lithosphere starts beneath adjoining continental plates at a 50° dip. The depth is an initial-condition choice; the force balance determines subsequent motion. A target below 1 selects whole connected trench arcs, so the realized share can exceed the target.";
    // The legacy velocity law is the only consumer of these two coefficients.
    for (const name of ["slab_pull", "ridge_push"]) {
      const field = $(name === "slab_pull" ? "slab-pull-field" : "ridge-push-field");
      $(name).disabled = reviewed;
      if (field) field.classList.toggle("inactive-field", reviewed);
    }
    $("force-coefficient-note").hidden = !reviewed;
    $("advanced-physics-controls").hidden = !reviewed;
    $("primordial-subduction-controls").hidden = !reviewed;
    $("primordial-ocean-controls").hidden = !reviewed;
    updateOceanAttachmentNote();
    const oceanStartup = $("primordial-ocean-enabled").checked ?
      (oceanAttachmentDraft.entries.length ? "Saved ocean regions are attached to continental plates as listed below; the remaining regions have independent plates. Their later motion is solved from forces." : `The ocean starts as ${Number($("primordial-ocean-count").value)} independent plates with young ridge boundaries. Their later motion is solved from forces.`) :
      "The primordial ocean starts as one plate.";
    const startup = $("primordial-subduction-enabled").checked ? (effective ?
      "Established initial trench arcs supply force from rest; ocean intake still requires actual convergence." :
      `Initial ocean subduction margins with ${Number($("primordial-slab-depth").value)} km deep inherited slabs. Later convergence follows the force balance.`) :
      "No inherited slabs or prescribed mantle forcing: startup may be nearly stationary.";
    const response = effective ? `Lite · effective trench force: ${Number($("effective-subduction-force").value)} × 10¹² N/m toward the trench on the incoming plate. No resolved slab or overriding-plate suction; plate ownership still determines which continent moves.` : $("subduction_response").value === "moving_hinge_v1" ? "Moving-trench slab gravity responds to relative plate motion and can move either plate; this is a constrained closure, not calibrated mantle suction." : "Historical fixed-trench slab response.";
    const phases = !effective && $("retained_phases").value === "thermal_v1" ? "Dense crust is retained with temperature-dependent phase evolution and drainage; a fresh world starts at its reference conductive-bath temperature." : "Historical immediate dense-crust foundering.";
    // Measured for the seeded field: mean = amplitude/2; the pattern is
    // unrelated (zero correlation) at about 3.15 x its length scale.
    const riftAmplitude = Number($("enhanced-rifting-amplitude").value);
    const riftDecorrelationKm = Math.round(Number($("enhanced-rifting-correlation").value) * 3.15 / 100) * 100;
    const rifting = $("enhanced-rifting-enabled").checked ?
      `Enhanced rifting: continental crust starts up to ${Math.round(riftAmplitude * 100)}% weaker (about ${(riftAmplitude * 50).toFixed(1).replace(/\.0$/, "")}% on average) in a seeded pattern that becomes unrelated over about ${riftDecorrelationKm} km (${riftDecorrelationKm / 2} px at 2 km/px); rift damage leaves a durable scar up to 50% weaker; rupture is judged on realized stretching of material links against a numerical breakup gate.` :
      "Historical rifting: rupture follows the coarse loading proxy.";
    $("physics-profile-note").textContent = reviewed ?
      `Force-balanced motion; conservative collision and crust budgets. ${oceanStartup} ${startup} ${response} ${phases} ${rifting} Regional basin correction is disabled; local isostasy remains active.` :
      "Legacy plate motion includes imposed background forcing and a response lag. The newer force-balance, weld and foundering options are not enabled by this setup.";
    const estimate = null; // Native geometry and raster frames have separate storage costs.
    $("grid-cost-note").hidden = !estimate;
    if (estimate) {
      $("grid-cost-note").textContent = `Approximate history storage: ~${(estimate * count / 501).toFixed(1)} GiB at this cadence. Actual storage depends on the starting world and its evolution.`;
    }
    const outputWidth = Number($("terrain-width").value), bytes = outputWidth * outputWidth / 2 * 4 / 1048576;
    $("terrain-resolution-note").textContent = `${outputWidth.toLocaleString()} × ${(outputWidth / 2).toLocaleString()} · ${(40030 / outputWidth).toFixed(1)} km per pixel at the equator · ${bytes.toLocaleString()} MiB of floating-point elevations.`;
    document.querySelector('output[for="terrain-detail"]').textContent = `${Number($("terrain-detail").value).toFixed(2)}×`;
  }

  function updateRecordedPhysicsNote(data, initial) {
    const note = $("recorded-physics-note");
    if (initial) {
      note.textContent = `Starting artwork: ${data.width} × ${data.height}. Plate fractures and initial conditions are seeded; this picture is not a tectonic prediction.`;
      return;
    }
    const cells = Number.isInteger(data.mesh_level) ? 20 * 4 ** data.mesh_level : null;
    const grid = `Recorded surface: ${data.width} × ${data.height}` + (cells ? ` · ${cells.toLocaleString()} tectonic control cells` : "");
    const physics = data.physics_profile_version === 1 ? "Force balance · revised physics · local isostasy; regional basin correction off" : "Legacy or individually configured physics; see saved experiment metadata";
    const effective = data.effective_subduction_version === 1;
    const responseVersion = data.subduction_response_version ?? data.physics_profile?.subduction_response_version;
    const phaseVersion = data.retained_dense_crust_version ?? data.physics_profile?.retained_dense_crust_version;
    const force = data.effective_subduction_diagnostics?.force_n_per_m;
    const response = effective ? ` · Lite · effective trench force${Number.isFinite(force) ? `, ${force / 1e12} × 10¹² N/m` : ""}, incoming plate only` : responseVersion === 1 ? " · moving-trench overriding-plate response" : data.physics_profile_version === 1 && (responseVersion === undefined || responseVersion === 0) ? " · historical fixed-trench response" : "";
    const phases = phaseVersion === 1 ? " · retained dense crust with thermal phase evolution" : data.physics_profile_version === 1 && (phaseVersion === undefined || phaseVersion === 0) ? " · historical immediate dense-crust foundering" : "";
    const riftGate = data.rift_mechanics?.rupture_strain_threshold;
    const rifting = data.enhanced_rifting_version === 1 ? ` · enhanced rifting: seeded inheritance, realized-strain rupture${Number.isFinite(riftGate) ? ` (numerical gate ${riftGate} logarithmic strain)` : ""}` : "";
    const timing = data.timestep_diagnostics;
    const retries = timing?.rejected_trials > 0 ? ` · ${timing.rejected_trials} rejected trial(s), whole timestep retried` : "";
    const primordial = data.primordial_subduction_diagnostics;
    const startup = effective ? " · declared initial trench arcs" : primordial?.enabled ? ` · initial ocean subduction, ${primordial.initial_slab_depth_km} km deep slabs` : "";
    const ocean = data.primordial_ocean_diagnostics;
    const attached = ocean?.continental_attachments || [];
    const oceanRegions = ocean?.water_region_count ?? ocean?.realized_plate_count ?? ocean?.plate_count;
    const independentOceanPlates = ocean?.independent_ocean_plate_count ?? oceanRegions - attached.length;
    const oceanStartup = ocean?.enabled ? (attached.length ? ` · initially ${oceanRegions} ocean regions: ${attached.length} attached to continental plates, ${independentOceanPlates} independent ocean plates` : ` · initially ${independentOceanPlates} ocean plates with seeded ridges`) : "";
    note.textContent = `${grid}. ${physics}${response}${phases}${rifting}${startup}${oceanStartup}${retries}.`;
  }

  function clampMapView() {
    const edge = .5 / mapView.zoom;
    // Longitude is periodic: horizontal pan wraps around the globe instead of stopping at
    // the +/-180 edge. Latitude is not, so vertical pan still clamps at the poles.
    mapView.cx = ((mapView.cx % 1) + 1) % 1;
    mapView.cy = Math.max(edge, Math.min(1 - edge, mapView.cy));
  }

  function updateMapZoomControls() {
    const available = Boolean(state.mode === "history" ? state.frame : state.initial);
    const zoom = globeView.active ? globeView.pose.zoom : mapView.zoom;
    $("map-zoom-out").disabled = !available || zoom <= 1;
    $("map-zoom-in").disabled = !available || zoom >= (globeView.active ? 4 : 16);
    $("map-zoom-fit").disabled = !available || zoom <= 1;
    $("map-zoom-label").textContent = `${zoom.toFixed(zoom % 1 ? 1 : 0)}×`;
    $("map-shell").classList.toggle("zoomed", available && mapView.zoom > 1);
    $("map-shell").classList.toggle("panning", !!mapView.drag);
  }

  function zoomMap(factor, fx = .5, fy = .5) {
    if (!(state.mode === "history" ? state.frame : state.initial)) return;
    if (globeView.active) { moveGlobe(globeView.pose.lon, globeView.pose.lat, globeView.pose.zoom * factor); return; }
    const next = Math.max(1, Math.min(16, mapView.zoom * factor));
    mapView.cx += (fx - .5) * (1 / mapView.zoom - 1 / next);
    mapView.cy += (fy - .5) * (1 / mapView.zoom - 1 / next);
    mapView.zoom = next; clampMapView(); updateMapZoomControls(); render();
  }

  const gosplPath = (job, endpoint) => `/api/gospl/${endpoint}?job_id=${encodeURIComponent(job.job_id)}`;
  const gosplRange = (job) => `${formatTime(job.start_myr)}–${formatTime(job.end_myr)} Myr elapsed`;
  const gosplSource = (job) => `${job.run_id === state.runId ? "Selected experiment" : "Different experiment"} · ${job.run_id} · ${gosplRange(job)} · ${savedOrientationNote(job)}`;
  const gosplNewest = (a, b) => String(b.created || b.job_id).localeCompare(String(a.created || a.job_id));

  function gosplFormError() {
    if (!state.runId || state.frames.length < 2) return "Save at least two epochs to export evolving history.";
    if (!Number.isInteger(gospl.startIndex) || !Number.isInteger(gospl.endIndex) || gospl.startIndex < 0 || gospl.endIndex >= state.frames.length || gospl.startIndex >= gospl.endIndex) return "Choose an ending epoch later than the starting epoch.";
    if (![6, 7, 8, 9].includes(Number($("gospl-subdivisions").value))) return "Choose a mesh level from 6 to 9.";
    const dt = Number($("gospl-dt").value);
    if (!Number.isSafeInteger(dt) || dt <= 0) return "Enter a positive whole-year goSPL time step.";
    const times = state.frames.slice(gospl.startIndex, gospl.endIndex + 1).map((frame) => Number(frame.time_myr) * 1000000);
    if (times.some((time, i) => !Number.isFinite(time) || Math.abs(time - Math.round(time)) > .001 || (i > 0 && Math.round(time) <= Math.round(times[i - 1])))) return "Saved epochs must increase at whole-year times.";
    if (times.some((time) => (Math.round(time) - Math.round(times[0])) % dt !== 0)) return "The goSPL time step must divide every selected saved interval exactly.";
    const rainfall = Number($("gospl-rainfall").value);
    if ($("gospl-rainfall").value === "" || !Number.isFinite(rainfall) || rainfall < 0 || rainfall > 20) return "Enter rainfall from 0 to 20 metres per year.";
    return "";
  }

  function reconcileGosplResult() {
    if (!gospl.selectionManual) gospl.result = gospl.jobs.filter((job) => job.state === "complete" && job.run_id === state.runId).sort(gosplNewest)[0] || null;
    const result = gospl.result;
    $("gospl-result").hidden = !result;
    $("download-gospl").disabled = !result;
    if (result) {
      $("gospl-result-title").textContent = `${gosplRange(result)} · ${Number(result.node_count).toLocaleString()} nodes`;
      $("gospl-result-source").textContent = gosplSource(result);
      $("gospl-result-source").classList.toggle("different-experiment", result.run_id !== state.runId || !sameOrientation(result.orientation || result.metadata?.orientation, outputOrientation()));
    }
    const jobs = gospl.jobs.filter((job) => job.state === "complete").sort(gosplNewest), picker = $("saved-gospl");
    picker.replaceChildren(new Option(jobs.length ? "Choose a completed history export…" : "No completed exports yet", ""));
    for (const job of jobs) picker.add(new Option(`${job.run_id === state.runId ? "This experiment" : "Other experiment"} · ${gosplRange(job)} · ${Number(job.node_count).toLocaleString()} nodes · ${job.run_id} · ${job.job_id.slice(-8)}`, job.job_id));
    picker.value = result?.job_id || ""; picker.disabled = !jobs.length;
    $("saved-gospl-count").textContent = jobs.length ? `(${jobs.length})` : "";
  }

  function updateGosplControls() {
    const changedRun = gospl.rangeRun !== state.runId;
    if (changedRun || gospl.rangeCount !== state.frames.length) {
      if (changedRun) { gospl.startIndex = 0; gospl.endManual = false; }
      const last = Math.max(0, state.frames.length - 1);
      gospl.startIndex = Math.min(gospl.startIndex, last);
      gospl.endIndex = gospl.endManual ? Math.min(gospl.endIndex, last) : last;
      gospl.rangeRun = state.runId; gospl.rangeCount = state.frames.length;
      for (const id of ["gospl-start", "gospl-end"]) {
        const select = $(id); select.replaceChildren();
        if (!state.frames.length) select.add(new Option("No saved history", ""));
        state.frames.forEach((frame, index) => select.add(new Option(`${formatTime(frame.time_myr)} Myr · frame ${index + 1}`, index)));
      }
      $("gospl-start").value = state.frames.length ? String(gospl.startIndex) : "";
      $("gospl-end").value = state.frames.length ? String(gospl.endIndex) : "";
      reconcileGosplResult();
    }
    const available = !!state.runId && state.frames.length >= 2 && !state.loadingRun && !state.runTransition && !state.starting && !globeOrientation.pending;
    const error = gosplFormError(), busy = gospl.starting || gospl.job?.state === "running";
    $("gospl-start").disabled = !available; $("gospl-end").disabled = !available;
    $("generate-gospl").disabled = !available || !!error || busy || !gospl.supported;
    $("generate-gospl").textContent = busy ? "Building history ZIP…" : "Build history ZIP";
    $("gospl-selected-source").textContent = state.runId && state.frames.length ? `Selected experiment · ${state.runId} · ${formatTime(state.frames[gospl.startIndex]?.time_myr)}–${formatTime(state.frames[gospl.endIndex]?.time_myr)} Myr elapsed · ${Math.max(0, gospl.endIndex - gospl.startIndex + 1)} saved epochs · ${orientationLabel(outputOrientation())}.` : "Save at least two epochs to export evolving history.";
    $("gospl-form-note").textContent = !gospl.supported ? "Restart the local engine to enable goSPL exports, then refresh this export list." : error;
    $("gospl-form-note").hidden = gospl.supported && !error;
    const job = gospl.starting ? gospl.pending : gospl.job;
    $("gospl-progress").hidden = !job;
    $("cancel-gospl").hidden = !busy;
    $("cancel-gospl").disabled = gospl.starting || gospl.cancelling || gospl.job?.state !== "running";
    $("cancel-gospl").textContent = gospl.cancelling ? "Cancelling…" : "Cancel export";
    if (job) {
      const progress = Math.max(0, Math.min(1, Number(job.progress || 0)));
      $("gospl-progress-bar").value = job.state === "complete" ? 1 : progress;
      $("gospl-progress-source").textContent = gosplSource(job);
      $("gospl-progress-source").classList.toggle("different-experiment", job.run_id !== state.runId || !sameOrientation(job.orientation || job.metadata?.orientation, outputOrientation()));
      $("gospl-progress-label").textContent = gospl.starting ? "Preparing the saved history…" : job.state === "error" ? (job.error || job.message || "The history export failed. Saved tectonic frames remain available.") : job.state === "cancelled" ? "Export cancelled. Completed exports and tectonic history remain available." : job.state === "complete" ? "History ZIP ready to download." : `${job.message || "Building history export"} · ${Math.round(progress * 100)}%`;
    }
    $("saved-gospl-status").textContent = gospl.notice || (gospl.listLoaded && !gospl.result ? "No completed export selected for this experiment. Previous packages remain in the list." : "");
    $("saved-gospl-status").hidden = !$("saved-gospl-status").textContent;
  }

  function applyGosplStatus(response) {
    const job = response?.job || response;
    if (!job?.job_id || !job.state) return;
    const previous = gospl.job;
    gospl.job = { ...job };
    captureTiming("gospl", job, job.job_id);
    gospl.jobs = [...gospl.jobs.filter((saved) => saved.job_id !== job.job_id), gospl.job].sort(gosplNewest);
    reconcileGosplResult(); updateGosplControls();
    if (job.state === "complete" && (previous?.job_id !== job.job_id || previous.state !== "complete")) toast(`goSPL history ZIP ready: ${gosplRange(job)} · experiment ${job.run_id}.`);
  }

  async function refreshGosplJobs() {
    const request = ++gospl.listRequest, previousJob = gospl.job;
    $("refresh-gospl").disabled = true;
    try {
      const response = await api("/api/gospl/list");
      if (request !== gospl.listRequest) return;
      const jobs = new Map((response.jobs || []).map((job) => [job.job_id, job]));
      for (const job of [gospl.result, gospl.job]) if (job) jobs.set(job.job_id, job);
      gospl.jobs = [...jobs.values()].sort(gosplNewest);
      gospl.listLoaded = true; gospl.supported = true; gospl.notice = "";
      if (gospl.job === previousJob && !gospl.starting && gospl.job?.state !== "running") gospl.job = gospl.jobs.find((job) => job.state === "running") || gospl.job;
      if (gospl.job) captureTiming("gospl", gospl.job, gospl.job.job_id);
      reconcileGosplResult(); updateGosplControls();
    } catch (error) {
      if (request === gospl.listRequest) { if (error.status === 404) gospl.supported = false; gospl.notice = "Saved goSPL exports could not be refreshed. Use ↻ to retry."; updateGosplControls(); }
    } finally { if (request === gospl.listRequest) $("refresh-gospl").disabled = false; }
  }

  async function pollGospl() {
    if (gospl.polling || gospl.starting || gospl.cancelling || !gospl.supported || gospl.job?.state !== "running") return;
    gospl.polling = true;
    const jobId = gospl.job.job_id, version = gospl.requestVersion;
    try { const response = await api(gosplPath(gospl.job, "status")); if (version === gospl.requestVersion && gospl.job?.job_id === jobId) { gospl.notice = ""; applyGosplStatus(response); } }
    catch (error) { gospl.notice = "Export progress could not be refreshed; retrying while the local engine is available."; updateGosplControls(); }
    finally { gospl.polling = false; }
  }

  async function generateGospl() {
    updateGosplControls();
    if ($("generate-gospl").disabled) return;
    const selected = { run_id: state.runId, start_index: gospl.startIndex, end_index: gospl.endIndex, subdivisions: Number($("gospl-subdivisions").value), dt_years: Number($("gospl-dt").value), rainfall_m_yr: Number($("gospl-rainfall").value), orientation: outputOrientation() };
    gospl.pending = { run_id: selected.run_id, start_myr: state.frames[selected.start_index].time_myr, end_myr: state.frames[selected.end_index].time_myr, orientation: selected.orientation };
    gospl.requestVersion = (gospl.requestVersion || 0) + 1;
    captureTiming("gospl", { elapsed_seconds: 0, eta_seconds: null, timing_state: "estimating", state: "running" }, "pending-gospl");
    gospl.starting = true; gospl.selectionManual = false; gospl.notice = ""; updateGosplControls();
    try { applyGosplStatus(await api("/api/gospl/start", selected)); }
    catch (error) { if (error.status === 404) gospl.supported = false; captureTiming("gospl", gospl.job, gospl.job?.job_id); gospl.notice = error.message; toast(error.message, true); }
    finally { gospl.starting = false; gospl.pending = null; updateGosplControls(); }
  }

  async function cancelGospl() {
    if (gospl.starting || gospl.cancelling || gospl.job?.state !== "running") return;
    const jobId = gospl.job.job_id;
    gospl.cancelling = true; gospl.requestVersion = (gospl.requestVersion || 0) + 1; updateGosplControls();
    try { const response = await api("/api/gospl/cancel", { job_id: jobId }); if (gospl.job?.job_id === jobId) applyGosplStatus(response); }
    catch (error) { toast(error.message, true); }
    finally { gospl.cancelling = false; updateGosplControls(); }
  }

  const terrainPath = (job, endpoint, extra = "") => `/api/terrain/${endpoint}?job_id=${encodeURIComponent(job.job_id)}${extra}`;
  const jobSize = (job) => `${Number(job.width).toLocaleString()} × ${Number(job.height || job.width / 2).toLocaleString()}`;
  const isGosplTerrain = (job) => job?.source_type === "gospl_result" || job?.metadata?.source_type === "gospl_result";
  const terrainTime = (job) => `${formatTime(job.time_myr)} Myr${isGosplTerrain(job) ? " solver time" : ""}`;
  const terrainMismatch = (job) => !isGosplTerrain(job) && (job.run_id !== state.runId || !sameOrientation(job.orientation || job.metadata?.orientation, outputOrientation()));
  function jobSource(job) {
    if (isGosplTerrain(job)) return `goSPL landscape · solver epoch ${job.source_result_epoch ?? job.frame} · ${job.source_path || "preserved solver surface"}`;
    const w = Number(job.source_width || job.metadata?.source_width), h = Number(job.source_height || job.metadata?.source_height);
    const grid = job.source_type === "native_history" ? "Native spherical mesh" : w && h ? `${w.toLocaleString()} × ${h.toLocaleString()} source grid` : "saved tectonic surface";
    return `${grid} · experiment ${job.run_id} · frame ${Number(job.frame) + 1}`;
  }

  function terrainSourceLabel(job) {
    if (isGosplTerrain(job)) return `${jobSource(job)} · imported globe orientation`;
    const margins = job.reconstruct_margins || job.metadata?.reconstruct_margins ? " · reconstructed continental margins" : "";
    return `${job.run_id === state.runId ? "Selected experiment" : "Different experiment"} · ${jobSource(job)}${margins} · ${savedOrientationNote(job)}`;
  }

  function reconcileTerrainExperiment() {
    if (!terrain.selectionManual) {
      terrain.resultRequest++; terrain.resultLoading = false;
      if (terrain.result?.run_id !== state.runId) {
        terrain.result = null; terrain.selectionId = null;
        $("terrain-result").hidden = true; $("terrain-thumbnail").removeAttribute("src");
      }
      const matching = [terrain.job, ...terrain.jobs].filter((job) => job?.state === "complete" && job.run_id === state.runId).sort((a, b) => String(b.created || b.job_id).localeCompare(String(a.created || a.job_id)))[0];
      if (!terrain.result && matching) selectSavedTerrain(matching.job_id);
    }
    if (terrain.result) $("terrain-result-source").textContent = terrainSourceLabel(terrain.result);
    $("terrain-result-source").classList.toggle("different-experiment", !!terrain.result && terrainMismatch(terrain.result));
    if (terrain.viewer) { $("terrain-view-source").textContent = terrainSourceLabel(terrain.viewer.job); $("terrain-view-source").classList.toggle("different-experiment", terrainMismatch(terrain.viewer.job)); }
    populateTerrainPicker(); updateTerrainResultLoading();
  }

  function updateTerrainControls() {
    const importing = $("terrain-source-mode").value === "gospl";
    const importReady = !!terrainImport.info && terrainImport.path === $("terrain-import-path").value.trim() && $("terrain-import-epoch").value !== "" && !terrainImport.loading && !terrainImport.browsing;
    const historyReady = state.mode === "history" && !!state.frame && state.frame.index === state.currentIndex && !!state.runId && !state.loadingRun && !globeOrientation.pending;
    const available = importing ? importReady : historyReady;
    const running = terrain.starting || terrain.job?.state === "running";
    $("terrain-import").hidden = !importing;
    $("terrain-native-options").hidden = importing;
    $("terrain-source-mode").disabled = running;
    $("terrain-import-path").disabled = running || terrainImport.browsing;
    $("browse-gospl-result").disabled = running || terrainImport.loading || terrainImport.browsing;
    $("inspect-gospl-result").disabled = running || terrainImport.loading || terrainImport.browsing || !$("terrain-import-path").value.trim();
    $("terrain-import-epoch").disabled = running || !terrainImport.info || terrainImport.loading;
    $("terrain-reconstruct-margins").disabled = running || !historyReady;
    $("generate-terrain").disabled = !available || running || !terrain.supported;
    $("generate-terrain").textContent = running ? "Generating terrain…" : importing ? "Render goSPL landscape" : "Generate selected time";
    $("terrain-width").disabled = running; $("terrain-detail").disabled = running;
    $("terrain-selected-source").textContent = importing ? importReady ? `Selected: goSPL solver epoch ${$("terrain-import-epoch").value} · original result orientation · detail ${Number($("terrain-detail").value).toFixed(2)}×.` : "Load a local goSPL landscape and select its solver epoch." : historyReady ? `Selected: ${formatTime(state.frame.time_myr)} Myr · native spherical surface · ${Number(state.frame.width).toLocaleString()} × ${Number(state.frame.height).toLocaleString()} review map · ${orientationLabel(outputOrientation())}.` : "Select a saved tectonic time to generate terrain.";
    if (!terrain.supported) $("terrain-selected-source").textContent = "Restart the local engine to enable detailed terrain generation.";
    $("terrain-progress").hidden = !running && !["error", "cancelled", "complete"].includes(terrain.job?.state);
    $("cancel-terrain").hidden = !running;
    $("cancel-terrain").disabled = terrain.starting || terrain.cancelling || terrain.job?.state !== "running";
    if (terrain.starting) $("terrain-progress-label").textContent = "Preparing the selected saved epoch…";
    else if (terrain.job?.state === "running") {
      const progress = Math.max(0, Math.min(1, Number(terrain.job.progress || 0)));
      $("terrain-progress-bar").value = progress;
      const phase = terrain.job.phase === "packaging" ? "Packaging terrain" : terrain.job.phase === "reading goSPL result" ? "Reading goSPL landscape" : "Sampling the surface";
      $("terrain-progress-label").textContent = `${phase} · ${jobSize(terrain.job)} at ${terrainTime(terrain.job)} · ${Math.round(progress * 100)}%`;
    } else if (terrain.job?.state === "error") $("terrain-progress-label").textContent = terrain.job.error || "Terrain generation stopped with an error. Try generating the selected time again.";
    else if (terrain.job?.state === "cancelled") $("terrain-progress-label").textContent = "Terrain generation stopped. Your tectonic history remains available.";
    else if (terrain.job?.state === "complete") { $("terrain-progress-label").textContent = `Terrain ready · ${jobSize(terrain.job)} at ${terrainTime(terrain.job)}`; $("terrain-progress-bar").value = 1; }
  }

  function applyTerrainStatus(job) {
    if (!job || !job.state) return;
    const previous = terrain.job;
    const changed = previous?.job_id !== job.job_id || previous?.state !== job.state;
    terrain.job = job;
    captureTiming("terrain", job, job.job_id);
    if (job.state === "complete" && job.job_id && changed && (job.run_id === state.runId || (isGosplTerrain(job) && $("terrain-source-mode").value === "gospl")) && (previous || !terrain.selectionManual)) displayTerrainResult(job, isGosplTerrain(job));
    if (changed && ["complete", "cancelled", "error"].includes(job.state)) refreshTerrainJobs();
    updateTerrainControls();
  }

  function populateTerrainPicker() {
    const complete = new Map(terrain.jobs.filter((job) => job.state === "complete").map((job) => [job.job_id, job]));
    for (const job of [terrain.result, terrain.job]) if (job?.state === "complete") complete.set(job.job_id, job);
    const jobs = [...complete.values()].sort((a, b) => String(b.created || b.job_id).localeCompare(String(a.created || a.job_id)));
    const select = $("saved-terrain");
    select.replaceChildren(new Option(jobs.length ? "Choose a saved terrain epoch…" : "No generated terrain yet", ""));
    for (const job of jobs) {
      const option = new Option(`${isGosplTerrain(job) ? "goSPL landscape" : job.run_id === state.runId ? "This experiment" : "Other experiment"} · ${terrainTime(job)} · ${jobSize(job)} · ${job.job_id.slice(-8)}`, job.job_id);
      option.title = `${jobSource(job)} · detail ${Number(job.detail ?? 1).toFixed(2)}× · ${job.created || job.job_id}`;
      select.add(option);
    }
    if (terrain.selectionId && complete.has(terrain.selectionId)) select.value = terrain.selectionId;
    select.disabled = !jobs.length;
    $("saved-terrain-count").textContent = jobs.length ? `(${jobs.length})` : "";
  }

  function updateTerrainResultLoading() {
    for (const id of ["review-terrain", "review-terrain-thumbnail", "terrain-download-png", "terrain-download-npy", "terrain-download-zip"]) $(id).disabled = terrain.resultLoading || !terrain.result;
    const empty = terrain.listLoaded && !terrain.result && !!state.runId;
    $("saved-terrain-status").hidden = !terrain.resultLoading && !empty;
    $("saved-terrain-status").textContent = terrain.resultLoading ? "Loading the saved terrain epoch…" : empty ? "No generated terrain for the selected experiment. Earlier builds remain in the saved-terrain list." : "";
  }

  function displayTerrainResult(job, manual = false) {
    terrain.resultRequest++;
    terrain.resultLoading = false;
    terrain.selectionManual = manual;
    terrain.selectionId = job.job_id;
    terrain.result = { ...job };
    $("terrain-result").hidden = false;
    $("terrain-thumbnail").src = terrainPath(job, "preview");
    $("terrain-thumbnail").alt = `Terrain relief at ${terrainTime(job)}, generated at ${jobSize(job)}`;
    $("terrain-result-title").textContent = `${jobSize(job)} · ${terrainTime(job)}`;
    $("terrain-result-source").textContent = terrainSourceLabel(job);
    $("terrain-result-source").classList.toggle("different-experiment", terrainMismatch(job));
    populateTerrainPicker(); updateTerrainResultLoading();
  }

  async function selectSavedTerrain(jobId, manual = false) {
    if (!jobId) { $("saved-terrain").value = terrain.selectionId || ""; return; }
    const previousManual = terrain.selectionManual;
    terrain.selectionManual = manual;
    const request = ++terrain.resultRequest;
    terrain.selectionId = jobId; terrain.resultLoading = true; updateTerrainResultLoading();
    try {
      const job = await api(`/api/terrain/status?job_id=${encodeURIComponent(jobId)}`);
      if (request !== terrain.resultRequest) return;
      if (job.state !== "complete") throw new Error("That terrain epoch is not available for review yet.");
      if (!manual && job.run_id !== state.runId) { reconcileTerrainExperiment(); return; }
      displayTerrainResult(job, manual);
    } catch (error) {
      if (request === terrain.resultRequest) { terrain.selectionManual = previousManual; terrain.selectionId = terrain.result?.job_id || null; populateTerrainPicker(); toast(error.message, true); }
    } finally { if (request === terrain.resultRequest) { terrain.resultLoading = false; updateTerrainResultLoading(); } }
  }

  async function refreshTerrainJobs() {
    const request = ++terrain.listRequest;
    $("refresh-terrain").disabled = true;
    try {
      const result = await api("/api/terrain/list");
      if (request !== terrain.listRequest) return;
      terrain.listLoaded = true;
      terrain.jobs = (result.jobs || []).filter((job) => job.state === "complete");
      populateTerrainPicker();
      const matching = terrain.jobs.find((job) => job.run_id === state.runId);
      if (!terrain.selectionId && matching) await selectSavedTerrain(matching.job_id);
      updateTerrainResultLoading();
    } catch (error) {
      if (request === terrain.listRequest) { $("saved-terrain-status").hidden = false; $("saved-terrain-status").textContent = "Saved terrain could not be refreshed. Use ↻ to retry."; }
    } finally { if (request === terrain.listRequest) $("refresh-terrain").disabled = false; }
  }

  async function pollTerrain() {
    if (terrain.polling || terrain.starting || !terrain.supported) return;
    terrain.polling = true;
    const epoch = terrain.epoch;
    try { const result = await api("/api/terrain/status"); if (epoch === terrain.epoch) applyTerrainStatus(result); }
    catch (error) {
      if (error.status === 404) { terrain.supported = false; updateTerrainControls(); $("terrain-selected-source").textContent = "Restart the local engine to enable detailed terrain generation."; }
    } finally { terrain.polling = false; }
  }

  async function generateTerrain() {
    if ($("generate-terrain").disabled) return;
    stopPlayback();
    const importing = $("terrain-source-mode").value === "gospl";
    const selected = importing ? { path: terrainImport.path, epoch_index: Number($("terrain-import-epoch").value), width: Number($("terrain-width").value), detail: Number($("terrain-detail").value) } : { run_id: state.runId, frame: state.currentIndex, width: Number($("terrain-width").value), detail: Number($("terrain-detail").value), orientation: outputOrientation(), reconstruct_margins: $("terrain-reconstruct-margins").checked };
    const epoch = ++terrain.epoch;
    captureTiming("terrain", { elapsed_seconds: 0, eta_seconds: null, timing_state: "estimating", state: "running" }, "pending-terrain");
    terrain.starting = true; updateTerrainControls();
    try {
      const result = await api(importing ? "/api/terrain/gospl" : "/api/terrain", selected);
      if (epoch !== terrain.epoch) return;
      applyTerrainStatus(result.job || result.status || result);
      toast(importing ? "Rendering the selected goSPL landscape. Its orientation is preserved." : "Generating terrain directly from the selected saved surface. You can continue reviewing its history.");
    } catch (error) { captureTiming("terrain", terrain.job, terrain.job?.job_id); toast(error.message, true); }
    finally { if (epoch === terrain.epoch) { terrain.starting = false; updateTerrainControls(); } }
  }

  function invalidateTerrainImport() {
    terrainImport.request++; terrainImport.info = null; terrainImport.path = null; terrainImport.loading = false;
    $("terrain-import-epoch").replaceChildren(new Option("Load results first", ""));
    $("terrain-import-status").textContent = "Load the selected output folder or descriptor to list its complete epochs.";
    updateTerrainControls();
  }

  async function inspectTerrainImport() {
    const path = $("terrain-import-path").value.trim();
    if (!path || terrainImport.loading) return;
    const request = ++terrainImport.request;
    terrainImport.info = null; terrainImport.path = null; terrainImport.loading = true;
    $("terrain-import-status").textContent = "Checking output descriptors and available epochs…";
    updateTerrainControls();
    try {
      const info = await api("/api/gospl/results/inspect", { path });
      if (request !== terrainImport.request || path !== $("terrain-import-path").value.trim()) return;
      terrainImport.info = info; terrainImport.path = path;
      const select = $("terrain-import-epoch"); select.replaceChildren();
      for (const item of info.epochs || []) select.add(new Option(`Epoch ${item.index} · ${formatTime(Number(item.time_years) / 1e6)} Myr solver time`, item.index));
      select.value = String(info.default_epoch);
      $("terrain-import-status").textContent = `${info.epochs.length} complete epoch${info.epochs.length === 1 ? "" : "s"} available. Elevation is read from all solver partitions. Imported results keep their own orientation.`;
    } catch (error) { if (request === terrainImport.request) { $("terrain-import-status").textContent = error.message; toast(error.message, true); } }
    finally { if (request === terrainImport.request) { terrainImport.loading = false; updateTerrainControls(); } }
  }

  async function browseTerrainImport() {
    if (terrainImport.browsing) return;
    terrainImport.browsing = true; updateTerrainControls();
    try {
      const result = await api("/api/gospl/results/browse", {});
      if (result.path) { $("terrain-import-path").value = result.path; invalidateTerrainImport(); await inspectTerrainImport(); }
    } catch (error) { toast(error.message, true); }
    finally { terrainImport.browsing = false; updateTerrainControls(); }
  }

  function terrainGeometry(view) {
    const rect = terrainCanvas.getBoundingClientRect(), fit = Math.min(rect.width / view.job.width, rect.height / view.job.height) * .98;
    return { width: rect.width, height: rect.height, fit, scale: fit * view.zoom };
  }

  function clampTerrainView(view) {
    const g = terrainGeometry(view), halfW = g.width / (2 * g.scale), halfH = g.height / (2 * g.scale);
    view.cx = halfW >= view.job.width / 2 ? view.job.width / 2 : Math.max(halfW, Math.min(view.job.width - halfW, view.cx));
    view.cy = halfH >= view.job.height / 2 ? view.job.height / 2 : Math.max(halfH, Math.min(view.job.height - halfH, view.cy));
  }

  function zoomTerrain(factor, px, py) {
    const view = terrain.viewer; if (!view) return;
    const g = terrainGeometry(view), x = px ?? g.width / 2, y = py ?? g.height / 2;
    const next = Math.max(1, Math.min(Math.max(16, 4 / g.fit), view.zoom * factor));
    const newScale = g.fit * next;
    view.cx += (x - g.width / 2) * (1 / g.scale - 1 / newScale);
    view.cy += (y - g.height / 2) * (1 / g.scale - 1 / newScale);
    view.zoom = next; clampTerrainView(view); renderTerrain();
  }

  function openTerrainViewer() {
    if (!terrain.result) return;
    const job = { ...terrain.result };
    const view = { job, zoom: 1, cx: job.width / 2, cy: job.height / 2, preview: null, tiles: new Map(), drag: null, sample: null, sampleRequest: 0, scheduled: false, error: null };
    terrain.viewer = view;
    $("terrain-view-title").textContent = `${jobSize(job)} terrain · ${terrainTime(job)}`;
    $("terrain-view-source").textContent = terrainSourceLabel(job);
    $("terrain-view-source").classList.toggle("different-experiment", terrainMismatch(job));
    $("terrain-sample").textContent = "Coordinates and elevation appear here when you click the map.";
    $("terrain-view-loading").hidden = false; $("terrain-view-loading").textContent = "Loading terrain preview…";
    if (!$("terrain-dialog").open) $("terrain-dialog").showModal();
    const preview = new Image();
    preview.onload = () => { if (terrain.viewer !== view) return; view.preview = preview; $("terrain-view-loading").hidden = true; renderTerrain(); };
    preview.onerror = () => { if (terrain.viewer !== view) return; view.error = "The preview could not be loaded. Close and reopen the viewer to retry."; renderTerrain(); };
    preview.src = terrainPath(job, "preview");
    renderTerrain();
  }

  function requestTerrainTiles(view, visible) {
    let running = [...view.tiles.values()].filter((entry) => entry.state === "loading").length;
    const keys = new Set(visible.map((tile) => tile.key));
    for (const [key, tile] of view.tiles) {
      if (view.tiles.size <= 48) break;
      if (!keys.has(key) && tile.state !== "loading") view.tiles.delete(key);
    }
    for (const tile of visible) {
      if (running >= 6) break;
      if (view.tiles.has(tile.key)) continue;
      const picture = new Image(), entry = { state: "loading", image: picture };
      view.tiles.set(tile.key, entry); running++;
      picture.onload = () => { entry.state = "ready"; if (terrain.viewer === view) renderTerrain(); };
      picture.onerror = () => { entry.state = "error"; if (terrain.viewer === view) renderTerrain(); };
      picture.src = terrainPath(view.job, "tile", `&x=${tile.x}&y=${tile.y}`);
    }
  }

  function renderTerrain() {
    const view = terrain.viewer;
    if (!view || view.scheduled || !$("terrain-dialog").open) return;
    view.scheduled = true;
    requestAnimationFrame(() => {
      view.scheduled = false;
      if (terrain.viewer !== view || !$("terrain-dialog").open) return;
      clampTerrainView(view);
      const g = terrainGeometry(view), ratio = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.round(g.width * ratio), height = Math.round(g.height * ratio);
      if (terrainCanvas.width !== width || terrainCanvas.height !== height) { terrainCanvas.width = width; terrainCanvas.height = height; }
      terrainContext.setTransform(ratio, 0, 0, ratio, 0, 0);
      terrainContext.fillStyle = "#07121a"; terrainContext.fillRect(0, 0, g.width, g.height);
      const ox = g.width / 2 - view.cx * g.scale, oy = g.height / 2 - view.cy * g.scale;
      terrainContext.imageSmoothingEnabled = g.scale < 1;
      if (view.preview) terrainContext.drawImage(view.preview, ox, oy, view.job.width * g.scale, view.job.height * g.scale);
      let ready = 0, failed = 0;
      const visible = [];
      if (g.scale * ratio >= .5) {
        const left = Math.max(0, Math.floor(-ox / g.scale / 512)), right = Math.min(Math.ceil(view.job.width / 512) - 1, Math.floor((g.width - ox) / g.scale / 512));
        const top = Math.max(0, Math.floor(-oy / g.scale / 512)), bottom = Math.min(Math.ceil(view.job.height / 512) - 1, Math.floor((g.height - oy) / g.scale / 512));
        for (let y = top; y <= bottom; y++) for (let x = left; x <= right; x++) {
          const key = `${x},${y}`; visible.push({ x, y, key });
          const entry = view.tiles.get(key);
          if (entry?.state === "ready") { terrainContext.drawImage(entry.image, ox + x * 512 * g.scale, oy + y * 512 * g.scale, entry.image.naturalWidth * g.scale, entry.image.naturalHeight * g.scale); ready++; }
          else if (entry?.state === "error") failed++;
        }
        visible.sort((a, b) => Math.hypot((a.x + .5) * 512 - view.cx, (a.y + .5) * 512 - view.cy) - Math.hypot((b.x + .5) * 512 - view.cx, (b.y + .5) * 512 - view.cy));
        requestTerrainTiles(view, visible);
      }
      if (view.sample) {
        const x = ox + (view.sample.x + .5) * g.scale, y = oy + (view.sample.y + .5) * g.scale;
        terrainContext.strokeStyle = "#fff0c3"; terrainContext.lineWidth = 1.5;
        terrainContext.beginPath(); terrainContext.arc(x, y, 7, 0, Math.PI * 2); terrainContext.moveTo(x - 12, y); terrainContext.lineTo(x + 12, y); terrainContext.moveTo(x, y - 12); terrainContext.lineTo(x, y + 12); terrainContext.stroke();
      }
      $("terrain-zoom-label").textContent = `${Math.round(g.scale * 100)}% pixels`;
      $("terrain-zoom-out").disabled = view.zoom <= 1;
      $("terrain-zoom-in").disabled = view.zoom >= Math.max(16, 4 / g.fit) - .01;
      $("terrain-tile-status").textContent = visible.length ? `${ready}/${visible.length} native tiles${failed ? ` · ${failed} unavailable` : ""}` : "World preview · zoom in for native terrain detail";
      if (view.error) { $("terrain-view-loading").hidden = false; $("terrain-view-loading").textContent = view.error; }
    });
  }

  async function sampleTerrain(clientX, clientY) {
    const view = terrain.viewer; if (!view) return;
    const rect = terrainCanvas.getBoundingClientRect(), g = terrainGeometry(view);
    const x = Math.floor(view.cx + (clientX - rect.left - g.width / 2) / g.scale), y = Math.floor(view.cy + (clientY - rect.top - g.height / 2) / g.scale);
    if (x < 0 || y < 0 || x >= view.job.width || y >= view.job.height) return;
    const request = ++view.sampleRequest;
    view.sample = { x, y }; renderTerrain();
    $("terrain-sample").textContent = `Sampling pixel ${x.toLocaleString()}, ${y.toLocaleString()}…`;
    try {
      const result = await api(terrainPath(view.job, "sample", `&x=${x}&y=${y}`));
      if (terrain.viewer !== view || request !== view.sampleRequest) return;
      $("terrain-sample").textContent = `${Math.abs(result.lat).toFixed(3)}° ${result.lat >= 0 ? "N" : "S"} · ${Math.abs(result.lon).toFixed(3)}° ${result.lon >= 0 ? "E" : "W"} · ${formatMetres(result.elevation_m)} · pixel ${x.toLocaleString()}, ${y.toLocaleString()}`;
    } catch (error) { if (terrain.viewer === view && request === view.sampleRequest) $("terrain-sample").textContent = error.message; }
  }

  $("config-form").addEventListener("submit", runSimulation);
  for (const id of ["slab_pull", "ridge_push", "erosion", "rift_strength", "brush-size"]) $(id).addEventListener("input", () => { updateRangeOutputs(); if (id === "brush-size") render(); });
  $("duration_myr").addEventListener("input", () => { updateResolutionNotes(); if (!state.frames.length) updateTimeline(); });
  $("snapshot_myr").addEventListener("input", updateResolutionNotes);
  $("mesh_level").addEventListener("change", updateResolutionNotes);
  $("physics_profile").addEventListener("change", updateResolutionNotes);
  $("subduction-force-law").addEventListener("change", selectSubductionForceLaw);
  $("effective-subduction-force").addEventListener("input", updateResolutionNotes);
  $("effective-force-rifting").addEventListener("change", () => {
    state.savedSubductionOptions ||= {};
    state.savedSubductionOptions.force_limit_rifting = {...(state.savedSubductionOptions.force_limit_rifting || {}), enabled: $("effective-force-rifting").checked};
  });
  for (const id of ["subduction_response", "retained_phases", "enhanced-rifting-enabled"]) $(id).addEventListener("change", updateResolutionNotes);
  for (const id of ["enhanced-rifting-amplitude", "enhanced-rifting-correlation"]) $(id).addEventListener("input", updateResolutionNotes);
  $("remove-ocean-attachments").addEventListener("click", removeOceanAttachments);
  $("primordial-subduction-enabled").addEventListener("change", () => { state.primordialDeclarationEdited = true; updateResolutionNotes(); });
  $("primordial-slab-depth").addEventListener("input", updateResolutionNotes);
  for (const id of ["primordial-margin-fraction", "primordial-selection-seed"]) $(id).addEventListener("input", () => { state.primordialDeclarationEdited = true; updateResolutionNotes(); });
  $("primordial-ocean-enabled").addEventListener("change", updateResolutionNotes);
  for (const id of ["primordial-ocean-count", "primordial-ocean-age-rate", "primordial-ocean-max-age"]) $(id).addEventListener("input", updateResolutionNotes);
  $("dt_myr").addEventListener("input", () => { updateRangeOutputs(); if (Number($("snapshot_myr").value) < Number($("dt_myr").value)) $("snapshot_myr").value = $("dt_myr").value; updateResolutionNotes(); });
  $("width").addEventListener("change", changeResolution);
  $("measure-toggle").addEventListener("click", () => setMeasureActive(!state.measure.active));
  $("measure-clear").addEventListener("click", () => { state.measure.points = []; updateMeasureReadout(); });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.measure.active) { setMeasureActive(false); }
  });
  $("regenerate").addEventListener("click", () => regenerate());
  $("clear-world").addEventListener("click", () => { if (!state.initial || isBusy()) return; beginEdit(); editHistory.changed = state.initial.crust.some(value => value !== 0) ? 1 : 0; state.initial.crust.fill(0); mapView.rasterData = null; finishEdit(); setMode("initial"); updateEvents(); toast("Starting world cleared to ocean. Undo restores your previous map."); });
  for (const [id, tool] of [["paint-tool", "brush"], ["fill-tool", "fill"]]) $(id).addEventListener("click", () => { state.editorTool = tool; updateEditorControls(); render(); });
  $("undo-edit").addEventListener("click", () => restoreEdit());
  $("redo-edit").addEventListener("click", () => restoreEdit(true));
  $("world-design-enabled").addEventListener("change", () => editWorldDesign(design => { design.enabled = $("world-design-enabled").checked; }));
  $("design-add").addEventListener("click", addDesignRegion);
  $("design-remove").addEventListener("click", () => editWorldDesign(design => { design.interventions = design.interventions.filter(row => row.id !== $("design-regions").value); }));
  $("design-preview").addEventListener("click", () => { try { designEditor.preview = readDesignRegion(true); render(); } catch (error) { toast(error.message, true); } });
  $("design-pick").addEventListener("click", () => { designEditor.picking = !designEditor.picking; updateWorldDesignControls(); });
  $("design-show").addEventListener("change", render);
  $("design-regions").addEventListener("change", updateWorldDesignControls);
  $("design-branch").addEventListener("click", branchWorldDesign);
  document.addEventListener("keydown", (event) => {
    if (state.mode !== "initial" || isBusy() || !(event.ctrlKey || event.metaKey) || event.target?.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(event.target?.tagName || "")) return;
    const key = event.key.toLowerCase();
    if (key === "z" || key === "y") { event.preventDefault(); restoreEdit(key === "y" || event.shiftKey); }
  });
  $("edit-world").addEventListener("click", () => { if (isBusy()) return; if (state.mode === "initial" && state.frames.length) showFrame(state.currentIndex); else { setMode("initial"); updateEvents(); } });
  for (const button of document.querySelectorAll(".brush-button")) button.addEventListener("click", () => { state.brush = Number(button.dataset.brush); for (const other of document.querySelectorAll(".brush-button")) { other.classList.toggle("selected", other === button); other.setAttribute("aria-pressed", String(other === button)); } });
  for (const button of document.querySelectorAll("[data-layer]")) button.addEventListener("click", () => { if (button.disabled) return; if (state.mode === "initial") { if (button.dataset.layer !== "crust") toast("The starting map shows crust materials. Run a simulation to inspect the other layers."); return; } state.layer = button.dataset.layer; setMode("history"); });
  $("show-boundaries").addEventListener("change", render); $("show-grid").addEventListener("change", render);
  $("show-slab-windows").addEventListener("change", render);
  $("show-lip").addEventListener("change", render);
  $("lip-add").addEventListener("click", addLipEvent);
  $("lip-remove").addEventListener("click", () => {
    lipDraft.events = lipDraft.events.filter(row => row.id !== $("lip-authored").value); updateLipDraft();
  });
  $("show-motion").addEventListener("change", () => { updateMotionLegend(); render(); });
  $("event-type").addEventListener("change", updateEvents); $("event-search").addEventListener("input", updateEvents);
  $("trench-select").addEventListener("change", () => { const id = Number($("trench-select").value); trenchReview.selectedId = Number.isSafeInteger(id) && id > 0 ? id : null; updateTrenchHistory(); });
  $("rift-select").addEventListener("change", () => { const id = Number($("rift-select").value); riftReview.selectedId = Number.isSafeInteger(id) && id > 0 ? id : null; updateRiftHistory(); });
  $("clear-inspection").addEventListener("click", clearInspection);
  $("refresh-inspection").addEventListener("click", () => { const selection = state.inspection?.selection; if (selection) loadInspection(selection.frame, selection.cell); });
  $("follow-live").addEventListener("change", () => { state.followLatest = $("follow-live").checked; if (state.followLatest && state.frames.length) { stopPlayback(); showFrame(state.frames.length - 1, { automatic: true }); } });
  $("jump-button").addEventListener("click", () => { if ($("jump-time").reportValidity()) visitTime(Number($("jump-time").value)); });
  $("jump-time").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); $("jump-button").click(); } });
  $("elevation-history").addEventListener("click", (event) => { const svg = $("elevation-history"), matrix = svg.getScreenCTM(); if (!matrix) return; const point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY; const x = point.matrixTransform(matrix.inverse()).x; visitTime(Math.max(0, Math.min(1, (x - 55) / 571)) * Number(svg.dataset.maxTime)); });
  $("elevation-history").addEventListener("keydown", (event) => { if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); stopPlayback(); showFrame(state.currentIndex + (event.key === "ArrowRight" ? 1 : -1)); } });
  $("timeline").addEventListener("input", () => { stopPlayback(); clearTimeout(state.scrubTimer); state.frameRequest++; if (state.inspectionPending) { state.inspectionRequest++; state.inspectionPending = false; updateInspection(); } state.currentIndex = Number($("timeline").value); updateTimeline(); state.scrubTimer = setTimeout(() => showFrame(state.currentIndex), 55); });
  $("play-button").addEventListener("click", togglePlayback);
  $("previous-frame").addEventListener("click", () => { stopPlayback(); showFrame(state.currentIndex - 1); });
  $("next-frame").addEventListener("click", () => { stopPlayback(); showFrame(state.currentIndex + 1); });
  $("pause-button").addEventListener("click", pauseOrResume);
  $("refresh-runs").addEventListener("click", refreshRuns); $("saved-runs").addEventListener("change", () => { $("load-run").disabled = isBusy() || !$("saved-runs").value; }); $("load-run").addEventListener("click", loadRun);
  $("export-initial").addEventListener("click", exportInitial); $("import-initial").addEventListener("click", () => $("initial-file").click()); $("initial-file").addEventListener("change", importInitial);
  $("download-heightmap").addEventListener("click", () => { download(orientedPath(`/api/heightmap?frame=${state.currentIndex}`)); toast("16-bit PNG: elevation in metres = pixel value − 12,000. The scale stays fixed across every snapshot."); });
  $("download-export").addEventListener("click", () => { download(orientedPath(`/api/export?frame=${state.currentIndex}`)); toast("Preparing the full experiment archive with all saved history and the selected snapshot's terrain exports."); });
  $("download-preview").addEventListener("click", () => download(orientedPath(`/api/preview?frame=${state.currentIndex}`)));
  for (const axis of ["yaw", "pitch", "roll"]) {
    for (const suffix of ["", "-slider"]) $(`orientation-${axis}${suffix}`).addEventListener("input", () => {
      if ($(`orientation-${axis}${suffix}`).value === "") return;
      const draft = state.mode === "initial" ? globeOrientation.draftInitial : globeOrientation.draftOutput;
      draft[axis] = Math.max(-180, Math.min(180, Number($(`orientation-${axis}${suffix}`).value) || 0));
      updateOrientationControls();
    });
    $(`orientation-${axis}`).addEventListener("change", updateOrientationControls);
  }
  $("orientation-apply").addEventListener("click", applyGlobeOrientation);
  $("orientation-reset").addEventListener("click", async () => {
    if (state.mode === "initial") { globeOrientation.draftInitial = normalizedOrientation(null); updateOrientationControls(); }
    else { globeOrientation.draftOutput = normalizedOrientation(null); updateOrientationControls(); await applyGlobeOrientation(); }
  });
  $("gospl-start").addEventListener("change", () => { gospl.startIndex = Number($("gospl-start").value); updateGosplControls(); });
  $("gospl-end").addEventListener("change", () => { gospl.endIndex = Number($("gospl-end").value); gospl.endManual = true; updateGosplControls(); });
  for (const id of ["gospl-subdivisions", "gospl-dt", "gospl-rainfall"]) $(id).addEventListener("input", updateGosplControls);
  $("generate-gospl").addEventListener("click", generateGospl);
  $("cancel-gospl").addEventListener("click", cancelGospl);
  $("refresh-gospl").addEventListener("click", refreshGosplJobs);
  $("saved-gospl").addEventListener("change", () => {
    const job = gospl.jobs.find((saved) => saved.job_id === $("saved-gospl").value && saved.state === "complete");
    if (job) { gospl.result = job; gospl.selectionManual = true; }
    reconcileGosplResult(); updateGosplControls();
  });
  $("download-gospl").addEventListener("click", () => { if (gospl.result?.state === "complete") download(gosplPath(gospl.result, "download")); });
  $("terrain-width").addEventListener("change", updateResolutionNotes);
  $("terrain-detail").addEventListener("input", () => { updateResolutionNotes(); updateTerrainControls(); });
  $("terrain-source-mode").addEventListener("change", () => { if ($("terrain-source-mode").value === "gospl") $("terrain-detail").value = "0"; updateResolutionNotes(); updateTerrainControls(); });
  $("terrain-import-path").addEventListener("input", invalidateTerrainImport);
  $("inspect-gospl-result").addEventListener("click", inspectTerrainImport);
  $("browse-gospl-result").addEventListener("click", browseTerrainImport);
  $("terrain-import-epoch").addEventListener("change", updateTerrainControls);
  $("saved-terrain").addEventListener("change", () => selectSavedTerrain($("saved-terrain").value, true));
  $("refresh-terrain").addEventListener("click", refreshTerrainJobs);
  $("generate-terrain").addEventListener("click", generateTerrain);
  $("cancel-terrain").addEventListener("click", async () => {
    if (!terrain.job?.job_id || terrain.starting || terrain.cancelling || terrain.job.state !== "running") return;
    terrain.cancelling = true; updateTerrainControls();
    try { await api("/api/terrain/cancel", { job_id: terrain.job.job_id }); await pollTerrain(); }
    catch (error) { toast(error.message, true); }
    finally { terrain.cancelling = false; updateTerrainControls(); }
  });
  $("review-terrain").addEventListener("click", openTerrainViewer);
  $("review-terrain-thumbnail").addEventListener("click", openTerrainViewer);
  for (const format of ["png", "npy", "zip"]) $(`terrain-download-${format}`).addEventListener("click", () => { if (terrain.result) download(terrainPath(terrain.result, "download", `&format=${format}`)); });
  $("close-terrain").addEventListener("click", () => $("terrain-dialog").close());
  $("terrain-dialog").addEventListener("close", () => { terrain.viewer = null; terrainCanvas.classList.remove("dragging"); });
  $("terrain-zoom-in").addEventListener("click", () => zoomTerrain(1.6));
  $("terrain-zoom-out").addEventListener("click", () => zoomTerrain(1 / 1.6));
  $("terrain-fit").addEventListener("click", () => { if (terrain.viewer) { terrain.viewer.zoom = 1; terrain.viewer.cx = terrain.viewer.job.width / 2; terrain.viewer.cy = terrain.viewer.job.height / 2; renderTerrain(); } });
  $("terrain-native").addEventListener("click", () => { if (terrain.viewer) zoomTerrain(1 / terrainGeometry(terrain.viewer).scale); });
  terrainCanvas.addEventListener("wheel", (event) => { event.preventDefault(); const rect = terrainCanvas.getBoundingClientRect(); zoomTerrain(Math.exp(-Math.max(-250, Math.min(250, event.deltaY)) * .003), event.clientX - rect.left, event.clientY - rect.top); }, { passive: false });
  terrainCanvas.addEventListener("pointerdown", (event) => {
    const view = terrain.viewer; if (!view || event.button !== 0) return;
    terrainCanvas.setPointerCapture(event.pointerId); terrainCanvas.classList.add("dragging");
    view.drag = { x: event.clientX, y: event.clientY, cx: view.cx, cy: view.cy, moved: false };
    event.preventDefault();
  });
  terrainCanvas.addEventListener("pointermove", (event) => {
    const view = terrain.viewer; if (!view?.drag) return;
    const dx = event.clientX - view.drag.x, dy = event.clientY - view.drag.y, g = terrainGeometry(view);
    if (Math.hypot(dx, dy) > 4) view.drag.moved = true;
    view.cx = view.drag.cx - dx / g.scale; view.cy = view.drag.cy - dy / g.scale; clampTerrainView(view); renderTerrain();
  });
  terrainCanvas.addEventListener("pointerup", (event) => {
    const view = terrain.viewer; if (!view?.drag) return;
    const click = !view.drag.moved; view.drag = null; terrainCanvas.classList.remove("dragging");
    if (click) sampleTerrain(event.clientX, event.clientY);
  });
  for (const type of ["pointercancel", "lostpointercapture"]) terrainCanvas.addEventListener(type, () => { if (terrain.viewer) terrain.viewer.drag = null; terrainCanvas.classList.remove("dragging"); });
  terrainCanvas.addEventListener("keydown", (event) => {
    const view = terrain.viewer; if (!view) return;
    if (["+", "="].includes(event.key)) { event.preventDefault(); zoomTerrain(1.6); }
    else if (event.key === "-") { event.preventDefault(); zoomTerrain(1 / 1.6); }
    else if (event.key === "0" || event.key === "Home") { event.preventDefault(); $("terrain-fit").click(); }
    else if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) { event.preventDefault(); const step = 80 / terrainGeometry(view).scale; view.cx += event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0; view.cy += event.key === "ArrowUp" ? -step : event.key === "ArrowDown" ? step : 0; clampTerrainView(view); renderTerrain(); }
    else if (event.key === "Enter") { event.preventDefault(); const rect = terrainCanvas.getBoundingClientRect(); sampleTerrain(rect.left + rect.width / 2, rect.top + rect.height / 2); }
  });
  $("about-button").addEventListener("click", () => $("about-dialog").showModal()); $("close-about").addEventListener("click", () => $("about-dialog").close()); $("about-dialog").addEventListener("click", (event) => { if (event.target === $("about-dialog")) { const r = $("about-dialog").getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) $("about-dialog").close(); } });
  $("fit-map").addEventListener("click", async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await $("map-shell").requestFullscreen(); } catch { toast("Full-screen viewing is unavailable in this browser."); } });
  $("exit-map-fullscreen").addEventListener("click", async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); } catch { toast("Use your browser's full-screen exit control."); } });
  window.addEventListener("resize", () => { render(); renderTerrain(); }); document.addEventListener("fullscreenchange", render);
  $("map-zoom-in").addEventListener("click", () => zoomMap(1.6));
  $("map-zoom-out").addEventListener("click", () => zoomMap(1 / 1.6));
  $("map-zoom-fit").addEventListener("click", () => { if (globeView.active) { moveGlobe(globeView.pose.lon, globeView.pose.lat, 1); return; } mapView.zoom = 1; mapView.cx = .5; mapView.cy = .5; updateMapZoomControls(); render(); });
  for (const [id, preferred] of [["view-flat", false], ["view-globe", true]]) $(id).addEventListener("click", () => {
    globeView.preferred = preferred; updateProjectionControls(); render();
    const url = new URL(location.href); if (preferred) url.searchParams.set("view", "globe"); else url.searchParams.delete("view");
    history.replaceState(null, "", url);
  });
  $("globe-home").addEventListener("click", () => moveGlobe(-60, 20, 1));
  for (const [from, to] of [["globe-play", "play-button"], ["globe-previous", "previous-frame"], ["globe-next", "next-frame"]]) $(from).addEventListener("click", () => $(to).click());
  $("globe-north").addEventListener("click", () => moveGlobe(globeView.pose.lon, 90, 1));
  $("globe-south").addEventListener("click", () => moveGlobe(globeView.pose.lon, -90, 1));
  globeCanvas.addEventListener("wheel", event => { event.preventDefault(); zoomMap(Math.exp(-Math.max(-250, Math.min(250, event.deltaY)) * .003)); }, {passive: false});
  globeCanvas.addEventListener("pointerdown", event => {
    if (event.button !== 0 || globeView.drag || !globePoint(event)) return;
    globeCanvas.focus({preventScroll: true}); globeCanvas.setPointerCapture(event.pointerId);
    globeView.drag = {id: event.pointerId, x: event.clientX, y: event.clientY, pose: {...globeView.pose}, moved: false};
    event.preventDefault();
  });
  globeCanvas.addEventListener("pointermove", event => {
    const drag = globeView.drag;
    if (drag) {
      if (drag.id !== event.pointerId) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      if (!drag.moved) return;
      const rect = globeCanvas.getBoundingClientRect(), scale = 180 / (Math.min(rect.width, rect.height) * drag.pose.zoom);
      globeCanvas.classList.add("dragging"); $("map-hover").hidden = true;
      moveGlobe(drag.pose.lon - dx * scale, drag.pose.lat + dy * scale); return;
    }
    const point = globePoint(event);
    if (point) showHover(point); else $("map-hover").hidden = true;
  });
  globeCanvas.addEventListener("pointerup", event => {
    const drag = globeView.drag; if (!drag || drag.id !== event.pointerId) return;
    globeView.drag = null; globeCanvas.classList.remove("dragging");
    if (!drag.moved) { const point = globePoint(event); if (point) inspectPoint(point); }
  });
  for (const event of ["pointercancel", "lostpointercapture"]) globeCanvas.addEventListener(event, () => { globeView.drag = null; globeCanvas.classList.remove("dragging"); });
  globeCanvas.addEventListener("pointerleave", () => { $("map-hover").hidden = true; });
  globeCanvas.addEventListener("keydown", event => {
    const {lon, lat, zoom} = globeView.pose, step = event.shiftKey ? 25 : 10;
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
      event.preventDefault(); moveGlobe(lon + (event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0), lat + (event.key === "ArrowUp" ? step : event.key === "ArrowDown" ? -step : 0));
    } else if (["+", "=", "-"].includes(event.key)) { event.preventDefault(); zoomMap(event.key === "-" ? 1 / 1.25 : 1.25); }
    else if (event.key === "Home") { event.preventDefault(); moveGlobe(-60, 20, 1); }
    else if (event.key === " ") { event.preventDefault(); $("play-button").click(); }
    else if (event.key === "Enter") { event.preventDefault(); const rect = globeCanvas.getBoundingClientRect(); const point = globePoint({clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2}); if (point) inspectPoint(point); }
  });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); const rect = canvas.getBoundingClientRect(); zoomMap(Math.exp(-Math.max(-250, Math.min(250, event.deltaY)) * .003), (event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height); }, { passive: false });
  canvas.addEventListener("pointerdown", (event) => {
    if (designEditor.picking && state.mode === "initial" && !isBusy() && event.button === 0) {
      const point = mapPoint(event); if (!point) return;
      $("design-lon").value = ((point.x + .5) / state.initial.width * 360 - 180).toFixed(3);
      $("design-lat").value = (90 - (point.y + .5) / state.initial.height * 180).toFixed(3);
      designEditor.picking = false; updateWorldDesignControls();
      try { designEditor.preview = readDesignRegion(true); render(); } catch (error) { toast(error.message, true); }
      event.preventDefault(); return;
    }
    // Any zoom: longitude wraps, so at full view a drag offsets the map east/west (latitude
    // clamps, so it cannot move vertically until zoomed in).
    if ((event.button === 1 || event.button === 0 && event.shiftKey) && !globeView.active) { canvas.setPointerCapture(event.pointerId); mapView.drag = { x: event.clientX, y: event.clientY, cx: mapView.cx, cy: mapView.cy }; updateMapZoomControls(); event.preventDefault(); return; }
    if (event.button !== 0) return; const point = mapPoint(event); if (!point) return;
    // Measuring takes precedence over both inspection and the paint brush, so a
    // measurement click can never edit the starting world by accident.
    if (state.measure.active) { addMeasurePoint(point); event.preventDefault(); return; }
    if (state.mode === "history") { inspectPoint(point); return; }
    if (isBusy()) return;
    beginEdit(); state.hover = point;
    if (state.editorTool === "fill") { editHistory.changed += fillMaterial(state.initial, point.y * state.initial.width + point.x, state.brush); mapView.rasterData = null; finishEdit(); render(); event.preventDefault(); return; }
    canvas.setPointerCapture(event.pointerId); state.drawing = true; state.lastPoint = null; updateEditorControls(); paintTo(point); event.preventDefault();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (mapView.drag) { const rect = canvas.getBoundingClientRect(); mapView.cx = mapView.drag.cx - (event.clientX - mapView.drag.x) / rect.width / mapView.zoom; mapView.cy = mapView.drag.cy - (event.clientY - mapView.drag.y) / rect.height / mapView.zoom; clampMapView(); render(); return; }
    const point = mapPoint(event); state.hover = point; showHover(point); if (state.drawing) paintTo(point); else if (state.mode === "initial") render();
  });
  function finishPainting() { if (state.drawing) { state.drawing = false; state.lastPoint = null; finishEdit(); } }
  canvas.addEventListener("pointerup", finishPainting); canvas.addEventListener("pointercancel", finishPainting); canvas.addEventListener("lostpointercapture", finishPainting);
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(type, () => { mapView.drag = null; updateMapZoomControls(); });
  canvas.addEventListener("pointerleave", () => { state.hover = null; $("map-hover").hidden = true; render(); });

  async function initialize() {
    updateRangeOutputs(); updateLegend(); updateResolutionNotes();
    try {
      const result = await api("/api/config"); fillConfig(result.defaults || result, { freshDefaults: true });
      await regenerate(false);
      const status = await api("/api/status");
      if (status.config) fillConfig(status.config);
      if (status.run_id) {
        state.initial = copyWorld(await api("/api/initial")); resetEditHistory();
      }
      await applyStatus(status); await refreshRuns();
      state.hasConnected = true;
    } catch (error) {
      $("map-empty").textContent = "The local engine is unavailable. Start the server, then refresh this page.";
      $("connection-status").textContent = "LOCAL ENGINE · OFFLINE";
      toast(error.message, true);
    }
    setInterval(poll, 1300);
    pollTerrain(); refreshTerrainJobs(); setInterval(pollTerrain, 1600);
    refreshGosplJobs(); setInterval(pollGospl, 1800);
    setInterval(updateTimingDisplays, 1000);
  }
  initialize();
})();
