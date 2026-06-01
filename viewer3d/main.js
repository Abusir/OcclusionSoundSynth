import * as THREE from "three";
import { OrbitControls } from "./node_modules/three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "./node_modules/three/examples/jsm/loaders/OBJLoader.js";

const canvas = document.getElementById("scene");
const statusEl = document.getElementById("status");
const sceneMetaEl = document.getElementById("sceneMeta");
const summaryEl = document.getElementById("summary");
const probeCanvas = document.getElementById("probePreview");
const probeYawInput = document.getElementById("probeYaw");
const probePitchInput = document.getElementById("probePitch");
const probeYawValue = document.getElementById("probeYawValue");
const probePitchValue = document.getElementById("probePitchValue");
const rgbPreview = document.getElementById("rgbPreview");
const rgbCaption = document.getElementById("rgbCaption");
const yawGridNote = document.getElementById("yawGridNote");
const sceneSelect = document.getElementById("sceneSelect");
const rgbPanel = document.getElementById("rgbPanel");
const yawPanel = document.getElementById("yawPanel");
const showOnlyImageScenes = document.getElementById("showOnlyImageScenes");

const DEFAULT_REPORT = "./data/latest_report.json";
const MANIFEST_PATH = "./data/scene_manifest.json";
const YAW_SWEEP_VALUES = [0, 90, 180, 270];

let report;
let latestReport;
let manifestEntries = [];
let scene;
let camera;
let renderer;
let controls;
let probeCamera;
let probeRenderer;
let probeFrustum;
let probeRay;
let receiverPoint;
let sourcePoint;
let cameraArrow;
let cameraFootprintGroup;
let loadedObject;
let alignedRgb = true;
const pressedKeys = new Set();

function setStatus(text) {
  statusEl.textContent = text;
}

function projectPath(path) {
  if (!path) return "";
  const normalized = String(path).replaceAll("\\", "/");
  const markers = ["/src/soundspaces_adapter/", "/soundspaces_adapter/", "/viewer3d/", "/src/viewer3d/", "/src/"];
  for (const marker of markers) {
    const idx = normalized.indexOf(marker);
    if (idx >= 0) {
      const sliced = normalized.slice(idx);
      if (sliced.startsWith("/soundspaces_adapter/")) return `/src${sliced}`;
      if (sliced.startsWith("/viewer3d/")) return sliced.replace("/viewer3d/", "./");
      return sliced;
    }
  }
  if (normalized.startsWith("src/soundspaces_adapter/")) return `/${normalized}`;
  if (normalized.startsWith("soundspaces_adapter/")) return `/src/${normalized}`;
  if (normalized.startsWith("viewer3d/")) return normalized.replace("viewer3d/", "./");
  return normalized;
}

function occToThree(xyz) {
  return new THREE.Vector3(xyz[0], xyz[2], xyz[1]);
}

function makeLabel(text, color = "#ffffff") {
  const canvasLabel = document.createElement("canvas");
  canvasLabel.width = 256;
  canvasLabel.height = 96;
  const ctx = canvasLabel.getContext("2d");
  ctx.clearRect(0, 0, canvasLabel.width, canvasLabel.height);
  ctx.font = "600 34px sans-serif";
  ctx.fillStyle = "rgba(0, 0, 0, 0.62)";
  ctx.fillRect(8, 18, 240, 52);
  ctx.fillStyle = color;
  ctx.fillText(text, 24, 55);
  const texture = new THREE.CanvasTexture(canvasLabel);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.7, 0.26, 1);
  return sprite;
}

function marker(position, color, label) {
  const group = new THREE.Group();
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 24, 16),
    new THREE.MeshStandardMaterial({ color, roughness: 0.48, metalness: 0.0 })
  );
  group.add(sphere);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.18, 0.012, 8, 48),
    new THREE.MeshBasicMaterial({ color })
  );
  ring.rotation.x = Math.PI / 2;
  group.add(ring);
  const sprite = makeLabel(label, color === 0x2b7de9 ? "#b9d5ff" : "#ffc2bd");
  sprite.position.set(0.35, 0.22, 0);
  group.add(sprite);
  group.position.copy(position);
  scene.add(group);
  return group;
}

function materialForName(name) {
  const lower = name.toLowerCase();
  if (lower.includes("floor")) return new THREE.MeshStandardMaterial({ color: 0x8d9085, roughness: 0.82, side: THREE.DoubleSide });
  if (lower.includes("ceiling")) return new THREE.MeshStandardMaterial({ color: 0xc7d0d8, roughness: 0.74, side: THREE.DoubleSide });
  if (lower.includes("obstacle")) return new THREE.MeshStandardMaterial({ color: 0x565149, roughness: 0.72, side: THREE.DoubleSide });
  return new THREE.MeshStandardMaterial({ color: 0xaeb8c1, roughness: 0.78, side: THREE.DoubleSide });
}

function applySceneMaterials(object) {
  object.traverse((child) => {
    if (!child.isMesh) return;
    const name = `${child.name || ""} ${child.material?.name || ""}`;
    child.material = materialForName(name);
    child.castShadow = true;
    child.receiveShadow = true;
  });
}

function fitCameraToObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1);
  camera.position.set(center.x + maxDim * 0.9, center.y + maxDim * 0.7, center.z + maxDim * 1.1);
  camera.near = 0.01;
  camera.far = maxDim * 12;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function cameraForwardVector() {
  const source = occToThree(report.source_xyz);
  const receiver = occToThree(report.receiver_xyz);
  const base = new THREE.Vector3().subVectors(source, receiver);
  base.y = 0;
  if (base.lengthSq() < 1e-8) base.set(1, 0, 0);
  base.normalize();
  if (report.rgb_alignment?.camera_footprint_is_reversed_180_deg) base.multiplyScalar(-1);
  const yawOffset = (report.config?.visual_yaw_offset_deg || 0) * Math.PI / 180;
  if (yawOffset) base.applyAxisAngle(new THREE.Vector3(0, 1, 0), yawOffset);
  return base.normalize();
}

function addCameraFootprint() {
  const receiver = occToThree(report.receiver_xyz);
  const forward = cameraForwardVector();
  cameraFootprintGroup = new THREE.Group();
  const arrowLength = 1.1;
  cameraArrow = new THREE.ArrowHelper(forward, receiver, arrowLength, 0xd8a51d, 0.22, 0.12);
  cameraFootprintGroup.add(cameraArrow);

  const fov = 90 * Math.PI / 180;
  const radius = 1.45;
  const left = forward.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), fov / 2);
  const right = forward.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), -fov / 2);
  const points = [
    receiver,
    receiver.clone().add(left.multiplyScalar(radius)),
    receiver.clone().add(right.multiplyScalar(radius)),
    receiver,
  ];
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xd8a51d }));
  cameraFootprintGroup.add(line);
  scene.add(cameraFootprintGroup);
}

function setupProbeCamera() {
  if (probeCamera) {
    setProbeAtReceiver();
    return;
  }
  probeRenderer = new THREE.WebGLRenderer({ canvas: probeCanvas, antialias: true });
  probeRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  probeCamera = new THREE.PerspectiveCamera(62, 4 / 3, 0.01, 1000);
  scene.add(probeCamera);
  probeFrustum = new THREE.CameraHelper(probeCamera);
  scene.add(probeFrustum);
  const material = new THREE.LineBasicMaterial({ color: 0x38c172 });
  probeRay = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, 1)]),
    material
  );
  scene.add(probeRay);
  setProbeAtReceiver();
}

function updateProbeUi() {
  probeYawValue.textContent = `${probeYawInput.value} deg`;
  probePitchValue.textContent = `${probePitchInput.value} deg`;
}

function setProbeAtReceiver() {
  const receiver = occToThree(report.receiver_xyz);
  probeCamera.position.copy(receiver).add(new THREE.Vector3(0, 0.06, 0));
  updateProbeCamera();
}

function setProbeFromMainView() {
  probeCamera.position.copy(camera.position);
  const direction = new THREE.Vector3();
  camera.getWorldDirection(direction);
  const yaw = Math.atan2(direction.z, direction.x) * 180 / Math.PI;
  const pitch = Math.asin(THREE.MathUtils.clamp(direction.y, -1, 1)) * 180 / Math.PI;
  probeYawInput.value = String(Math.round(yaw));
  probePitchInput.value = String(Math.round(pitch));
  updateProbeCamera();
}

function updateProbeCamera() {
  if (!probeCamera) return;
  updateProbeUi();
  const yaw = THREE.MathUtils.degToRad(Number(probeYawInput.value));
  const pitch = THREE.MathUtils.degToRad(Number(probePitchInput.value));
  const forward = new THREE.Vector3(
    Math.cos(yaw) * Math.cos(pitch),
    Math.sin(pitch),
    Math.sin(yaw) * Math.cos(pitch)
  ).normalize();
  probeCamera.lookAt(probeCamera.position.clone().add(forward));
  probeCamera.updateProjectionMatrix();
  probeFrustum.update();
  const points = [probeCamera.position.clone(), probeCamera.position.clone().add(forward.clone().multiplyScalar(1.1))];
  probeRay.geometry.dispose();
  probeRay.geometry = new THREE.BufferGeometry().setFromPoints(points);
}

function setReceiverView() {
  const receiver = occToThree(report.receiver_xyz);
  const source = occToThree(report.source_xyz);
  const dir = new THREE.Vector3().subVectors(source, receiver);
  dir.y = 0;
  if (dir.lengthSq() < 1e-8) dir.set(1, 0, 0);
  dir.normalize();
  camera.position.copy(receiver).add(new THREE.Vector3(0, 0.08, 0));
  controls.target.copy(receiver.clone().add(dir));
  controls.update();
}

function setCameraFootprintView() {
  const receiver = occToThree(report.receiver_xyz);
  const dir = cameraForwardVector();
  camera.position.copy(receiver).add(new THREE.Vector3(0, 0.08, 0));
  controls.target.copy(receiver.clone().add(dir));
  controls.update();
}

function updateRgbPreview() {
  const rawPath = report.rgb_raw?.path;
  const alignedPath = report.rgb?.path;
  const selected = alignedRgb ? alignedPath : rawPath;
  if (selected) {
    rgbPanel.classList.remove("is-hidden");
    rgbPreview.style.display = "block";
    rgbPreview.src = projectPath(selected);
    rgbCaption.textContent = alignedRgb ? "Aligned RGB, mirrored for top-down footprint" : "Raw Habitat RGB";
    document.getElementById("toggleRgbBtn").textContent = alignedRgb ? "Raw RGB" : "Aligned RGB";
  } else {
    rgbPanel.classList.add("is-hidden");
    rgbPreview.removeAttribute("src");
    rgbPreview.style.display = "none";
    document.getElementById("toggleRgbBtn").textContent = "No RGB";
  }
}

function yawSweepImagePath(yaw, raw = false) {
  const suffix = raw ? "rgb_raw" : "rgb";
  if (report.soundspaces_yaw_sweep_root) {
    return `${projectPath(report.soundspaces_yaw_sweep_root)}/yaw_${yaw}/figures/${(report.case || "los").toLowerCase()}_${report.scene_id}_${suffix}.png`;
  }
  const sceneId = report.scene_id;
  const caseName = (report.case || "los").toLowerCase();
  return `./data/yaw_calibration/yaw_${yaw}/figures/${caseName}_${sceneId}_${suffix}.png`;
}

function loadYawSweepGrid() {
  if (!report.rgb?.path) {
    yawPanel.classList.add("is-hidden");
    for (const yaw of YAW_SWEEP_VALUES) {
      const img = document.getElementById(`yaw${yaw}`);
      const caption = document.getElementById(`yaw${yaw}Caption`);
      img.removeAttribute("src");
      caption.textContent = `yaw ${yaw}`;
    }
    return;
  }
  yawPanel.classList.remove("is-hidden");
  let available = 0;
  for (const yaw of YAW_SWEEP_VALUES) {
    const img = document.getElementById(`yaw${yaw}`);
    const caption = document.getElementById(`yaw${yaw}Caption`);
    const aligned = yawSweepImagePath(yaw, false);
    const raw = yawSweepImagePath(yaw, true);
    img.src = aligned;
    img.onerror = () => {
      img.src = raw;
      caption.textContent = `yaw ${yaw} / raw`;
    };
    img.onload = () => {
      available += 1;
      caption.textContent = `yaw ${yaw} / aligned`;
      yawGridNote.textContent = "These are the actual SoundSpaces renders from the yaw calibration sweep.";
    };
  }
  window.setTimeout(() => {
    if (available === 0) {
      yawGridNote.textContent = "Yaw sweep images are not present yet. Run scripts/visualize_scene.py or a SoundSpaces visual yaw sweep to populate this grid.";
    }
  }, 1200);
}

async function loadReport() {
  const params = new URLSearchParams(window.location.search);
  const reportPath = params.get("report") || DEFAULT_REPORT;
  const response = await fetch(reportPath);
  if (!response.ok) throw new Error(`Cannot fetch report: ${reportPath}`);
  return response.json();
}

async function loadManifest() {
  const response = await fetch(MANIFEST_PATH);
  if (!response.ok) return [];
  const manifest = await response.json();
  return Array.isArray(manifest.entries) ? manifest.entries : [];
}

function populateSceneSelect() {
  const selectedBefore = sceneSelect.value;
  sceneSelect.innerHTML = "";
  const latestOption = document.createElement("option");
  latestOption.value = "__latest__";
  latestOption.textContent = "Latest SoundSpaces Report";
  sceneSelect.appendChild(latestOption);
  const filteredEntries = showOnlyImageScenes.checked
    ? manifestEntries.filter((entry) => entry.has_soundspaces_images)
    : manifestEntries;
  for (const entry of filteredEntries) {
    const option = document.createElement("option");
    option.value = String(entry.scene_index);
    const marker = entry.has_soundspaces_images ? "[IMG] " : "";
    option.textContent = `${marker}${entry.scene_index}: ${entry.scene_id}`;
    sceneSelect.appendChild(option);
  }
  if ([...sceneSelect.options].some((option) => option.value === selectedBefore)) {
    sceneSelect.value = selectedBefore;
  }
}

function clearLoadedScene() {
  for (const item of [loadedObject, receiverPoint, sourcePoint, cameraFootprintGroup]) {
    if (item) scene.remove(item);
  }
  loadedObject = null;
  receiverPoint = null;
  sourcePoint = null;
  cameraArrow = null;
  cameraFootprintGroup = null;
}

async function loadCurrentView(data) {
  report = data;
  alignedRgb = true;
  clearLoadedScene();
  const loader = new OBJLoader();
  const meshPath = projectPath(report.audio_visual_mesh || report.geometry?.obj || "");
  setStatus(`Loading ${meshPath}`);
  loadedObject = await loader.loadAsync(meshPath);
  applySceneMaterials(loadedObject);
  scene.add(loadedObject);

  receiverPoint = marker(occToThree(report.receiver_xyz), 0x2b7de9, "receiver");
  sourcePoint = marker(occToThree(report.source_xyz), 0xdd4538, "source");
  addCameraFootprint();
  setupProbeCamera();
  fitCameraToObject(loadedObject);

  const rirText = Array.isArray(report.rir_shape) ? ` / RIR ${report.rir_shape.join(" x ")}` : "";
  sceneMetaEl.textContent = `${report.scene_id} / ${report.case?.toUpperCase?.() || "CASE"}${rirText}`;
  summaryEl.textContent = `Receiver ${report.receiver_xyz.map((v) => v.toFixed(2)).join(", ")} m. Source ${report.source_xyz.map((v) => v.toFixed(2)).join(", ")} m.`;
  if (report.has_soundspaces_images) {
    summaryEl.textContent += ` SoundSpaces images available${report.soundspaces_source ? ` from ${report.soundspaces_source}` : ""}.`;
  }
  setStatus(
    report.rgb?.path
      ? `RGB alignment: mirror=${Boolean(report.rgb_alignment?.rgb_output_is_mirrored_horizontally)}, footprint +180=${Boolean(report.rgb_alignment?.camera_footprint_is_reversed_180_deg)}`
      : "Browser-only catalog mode. Use Receiver View and Probe Camera to inspect the camera direction."
  );
  updateRgbPreview();
  loadYawSweepGrid();
}

async function init() {
  latestReport = await loadReport();
  manifestEntries = await loadManifest();

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x141719);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;

  camera = new THREE.PerspectiveCamera(62, 1, 0.01, 1000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  const hemi = new THREE.HemisphereLight(0xe8f2ff, 0x35302a, 1.9);
  scene.add(hemi);
  const directional = new THREE.DirectionalLight(0xffffff, 1.6);
  directional.position.set(2.5, 5.5, 3.5);
  directional.castShadow = true;
  scene.add(directional);

  const grid = new THREE.GridHelper(8, 16, 0x36434d, 0x283038);
  grid.position.y = -0.002;
  scene.add(grid);
  populateSceneSelect();
  await loadCurrentView(latestReport);

  document.getElementById("overviewBtn").addEventListener("click", () => fitCameraToObject(loadedObject));
  document.getElementById("receiverBtn").addEventListener("click", setReceiverView);
  document.getElementById("cameraBtn").addEventListener("click", setCameraFootprintView);
  document.getElementById("toggleRgbBtn").addEventListener("click", () => {
    if (!report.rgb?.path && !report.rgb_raw?.path) return;
    alignedRgb = !alignedRgb;
    updateRgbPreview();
  });
  document.getElementById("probeReceiverBtn").addEventListener("click", setProbeAtReceiver);
  document.getElementById("syncMainBtn").addEventListener("click", setProbeFromMainView);
  probeYawInput.addEventListener("input", updateProbeCamera);
  probePitchInput.addEventListener("input", updateProbeCamera);
  window.addEventListener("keydown", (event) => pressedKeys.add(event.key.toLowerCase()));
  window.addEventListener("keyup", (event) => pressedKeys.delete(event.key.toLowerCase()));
  showOnlyImageScenes.addEventListener("change", () => {
    populateSceneSelect();
    if (sceneSelect.value !== "__latest__" && ![...sceneSelect.options].some((option) => option.value === sceneSelect.value)) {
      sceneSelect.value = "__latest__";
      loadCurrentView(latestReport);
    }
  });
  sceneSelect.addEventListener("change", async () => {
    if (sceneSelect.value === "__latest__") {
      await loadCurrentView(latestReport);
      return;
    }
    const next = manifestEntries.find((entry) => String(entry.scene_index) === sceneSelect.value);
    if (next) await loadCurrentView(next);
  });

  window.addEventListener("resize", resize);
  resize();
  animate();
}

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = Math.max(rect.width / Math.max(rect.height, 1), 0.1);
  camera.updateProjectionMatrix();
  const probeRect = probeCanvas.getBoundingClientRect();
  if (probeRenderer && probeCamera && probeRect.width > 0 && probeRect.height > 0) {
    probeRenderer.setSize(probeRect.width, probeRect.height, false);
    probeCamera.aspect = probeRect.width / probeRect.height;
    probeCamera.updateProjectionMatrix();
    probeFrustum?.update();
  }
}

function animate() {
  requestAnimationFrame(animate);
  updateKeyboardMotion();
  controls.update();
  renderer.render(scene, camera);
  if (probeRenderer && probeCamera) {
    const prevFrustumVisible = probeFrustum?.visible ?? false;
    const prevRayVisible = probeRay?.visible ?? false;
    if (probeFrustum) probeFrustum.visible = false;
    if (probeRay) probeRay.visible = false;
    probeRenderer.render(scene, probeCamera);
    if (probeFrustum) probeFrustum.visible = prevFrustumVisible;
    if (probeRay) probeRay.visible = prevRayVisible;
  }
}

function updateKeyboardMotion() {
  if (pressedKeys.size === 0) return;
  const speed = pressedKeys.has("shift") ? 0.08 : 0.035;
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0;
  if (forward.lengthSq() > 1e-8) forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const move = new THREE.Vector3();
  if (pressedKeys.has("w")) move.add(forward);
  if (pressedKeys.has("s")) move.sub(forward);
  if (pressedKeys.has("a")) move.add(right);
  if (pressedKeys.has("d")) move.sub(right);
  if (pressedKeys.has("e")) move.y += 1;
  if (pressedKeys.has("q")) move.y -= 1;
  if (move.lengthSq() <= 1e-8) return;
  move.normalize().multiplyScalar(speed);
  camera.position.add(move);
  controls.target.add(move);
}

init().catch((error) => {
  console.error(error);
  setStatus(error.message);
});
