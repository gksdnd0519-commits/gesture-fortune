import {
  FilesetResolver,
  GestureRecognizer,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";
import { FORTUNES, LUCKY_COLORS, LUCKY_ITEMS } from "./fortunes.js";

const HOLD_MS = 1500; // 제스처를 이만큼 유지하면 운세 뽑기
const RELEASE_MS = 600; // 뽑은 뒤 제스처를 이만큼 풀어야 다시 뽑을 수 있음
const THRESHOLD = 0.8; // 내 제스처로 인정할 최소 확률 (Python 쪽과 동일)
const BUILTIN = ["Victory", "Thumb_Up", "Open_Palm", "Closed_Fist", "Pointing_Up", "ILoveYou", "Thumb_Down"];
const GRADE_COLORS = { 대길: "#ffd166", 길: "#7bd88f", 중길: "#6ec6ff", 소길: "#5fd4c4", 평: "#c0c4cc", 주의: "#ff7a7a" };

const $ = (sel) => document.querySelector(sel);
const video = $("#video");
const canvas = $("#view");
const ctx = canvas.getContext("2d");

// ---------------- 내 제스처 분류기 (Python 의 gesture_features.py + MLP 와 같은 계산) ----------------

export function toFeatures(world, handName) {
  const pts = world.map((p) => [p.x - world[0].x, p.y - world[0].y, p.z - world[0].z]);
  if (handName === "Left") pts.forEach((p) => (p[0] *= -1));
  const scale = Math.max(...pts.map(([x, y, z]) => Math.hypot(x, y, z)));
  return pts.flat().map((v) => (scale > 0 ? v / scale : v));
}

export function predictProba(model, feat) {
  let a = feat.map((v, i) => (v - model.mean[i]) / model.scale[i]);
  model.coefs.forEach((w, layer) => {
    const b = model.intercepts[layer];
    const z = b.map((bias, j) => a.reduce((sum, v, i) => sum + v * w[i][j], bias));
    a = layer < model.coefs.length - 1 ? z.map((v) => Math.max(0, v)) : z;
  });
  if (model.out_activation === "logistic") {
    const p = 1 / (1 + Math.exp(-a[0]));
    return [1 - p, p];
  }
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const total = e.reduce((s, v) => s + v, 0);
  return e.map((v) => v / total);
}

function classifyHands(result, model) {
  const handedness = result.handedness ?? result.handednesses;
  return result.landmarks.map((points, i) => {
    const hand = handedness[i][0].categoryName;
    const builtin = result.gestures[i][0];
    let label = builtin.categoryName;
    let score = builtin.score;
    let custom = false;
    if (model) {
      const proba = predictProba(model, toFeatures(result.worldLandmarks[i], hand));
      const best = proba.indexOf(Math.max(...proba));
      if (proba[best] >= THRESHOLD && model.labels[best] !== "none") {
        [label, score, custom] = [model.labels[best], proba[best], true];
      }
    }
    // 화면을 좌우반전해서 보여주므로 왼손/오른손 라벨도 반전
    return { points, label, score, custom, side: hand === "Left" ? "오른손" : "왼손" };
  });
}

// ---------------- 화면 그리기 ----------------

function drawHands(hands, target) {
  const w = canvas.width;
  const h = canvas.height;
  for (const hand of hands) {
    const pts = hand.points.map((p) => [p.x * w, p.y * h]);
    const hit = matches(hand, target);
    ctx.strokeStyle = hit ? "#ffd166" : "rgba(255,255,255,0.85)";
    ctx.lineWidth = 3;
    for (const { start, end } of GestureRecognizer.HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(...pts[start]);
      ctx.lineTo(...pts[end]);
      ctx.stroke();
    }
    ctx.fillStyle = hit ? "#ffd166" : "#ff5b7f";
    for (const [x, y] of pts) {
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    const x0 = Math.min(...pts.map((p) => p[0]));
    const y0 = Math.max(Math.min(...pts.map((p) => p[1])) - 14, 28);
    ctx.font = "bold 24px 'Malgun Gothic', sans-serif";
    ctx.lineWidth = 5;
    ctx.strokeStyle = "rgba(0,0,0,0.8)";
    const text = `${hand.label} ${hand.score.toFixed(2)}`;
    ctx.strokeText(text, x0, y0);
    ctx.fillStyle = hand.custom ? "#7bd88f" : "#ffd166";
    ctx.fillText(text, x0, y0);
  }
}

function drawHoldRing(hand, progress) {
  const xs = hand.points.map((p) => p.x * canvas.width);
  const ys = hand.points.map((p) => p.y * canvas.height);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const r = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 2 + 24;
  ctx.lineWidth = 8;
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = "#ffd166";
  ctx.beginPath();
  ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
  ctx.stroke();
}

// ---------------- 운세 뽑기 ----------------

const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
let lastFortune = -1;
let drawCount = 0;

function matches(hand, target) {
  return target === "__any__" ? hand.custom : hand.label === target;
}

function reveal(gestureLabel) {
  let idx;
  do idx = Math.floor(Math.random() * FORTUNES.length);
  while (idx === lastFortune && FORTUNES.length > 1);
  lastFortune = idx;
  const fortune = FORTUNES[idx];
  const [colorName, colorHex] = pick(LUCKY_COLORS);
  drawCount += 1;

  $("#card-empty").hidden = true;
  $("#card-body").hidden = false;
  $("#grade").textContent = fortune.grade;
  $("#grade").style.color = GRADE_COLORS[fortune.grade];
  $("#fortune-text").textContent = fortune.text;
  $("#lucky-color").textContent = colorName;
  $("#lucky-swatch").style.background = colorHex;
  $("#lucky-number").textContent = Math.floor(Math.random() * 99) + 1;
  $("#lucky-item").textContent = pick(LUCKY_ITEMS);
  $("#card-meta").textContent = `'${gestureLabel}' 제스처로 뽑은 ${drawCount}번째 운세 · No.${idx + 1}`;

  const card = $("#card");
  card.classList.remove("reveal");
  void card.offsetWidth; // 애니메이션 다시 재생
  card.classList.add("reveal");
}

// ---------------- 메인 루프 ----------------

let model = null;
let recognizer = null;
let holdStart = null;
let locked = false;
let lastMatch = 0;
let lastVideoTime = -1;

function setStatus(text) {
  $("#status").textContent = text;
}

function targetName(target) {
  return target === "__any__" ? "내 제스처 아무거나" : `'${target}'`;
}

function updateHold(hands, now) {
  const target = $("#target").value;
  const hit = hands.find((h) => matches(h, target));
  const bar = $("#hold-bar");

  if (!hit) {
    holdStart = null;
    bar.style.width = "0%";
    if (locked && now - lastMatch > RELEASE_MS) locked = false;
    setStatus(hands.length ? `${targetName(target)} 제스처를 해 보세요` : "카메라에 손을 보여주세요");
    return;
  }
  lastMatch = now;
  if (locked) {
    bar.style.width = "0%";
    setStatus("손을 내렸다가 다시 하면 또 뽑을 수 있어요");
    return;
  }
  holdStart ??= now;
  const progress = Math.min((now - holdStart) / HOLD_MS, 1);
  bar.style.width = `${progress * 100}%`;
  drawHoldRing(hit, progress);
  setStatus("그대로 유지하세요… 운세를 뽑는 중 🔮");
  if (progress >= 1) {
    reveal(hit.label);
    locked = true;
    holdStart = null;
  }
}

function frame() {
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    // 학습 때(Python)처럼 좌우반전한 화면으로 인식해야 특징이 일치함
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    const now = performance.now();
    const result = recognizer.recognizeForVideo(canvas, now);
    const hands = classifyHands(result, model);
    drawHands(hands, $("#target").value);
    updateHold(hands, now);
  }
  requestAnimationFrame(frame);
}

async function loadModel() {
  try {
    const res = await fetch("gesture_model.json", { cache: "no-store" });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

function fillTargets() {
  const select = $("#target");
  const custom = model ? model.labels.filter((l) => l !== "none") : [];
  const groups = [];
  if (custom.length) {
    groups.push(`<optgroup label="내 제스처">
      <option value="__any__">아무거나 (${custom.join(", ")})</option>
      ${custom.map((l) => `<option value="${l}">${l}</option>`).join("")}
    </optgroup>`);
  }
  groups.push(`<optgroup label="기본 제스처">${BUILTIN.map((l) => `<option value="${l}">${l}</option>`).join("")}</optgroup>`);
  select.innerHTML = groups.join("");
  $("#model-info").textContent = custom.length
    ? `학습된 내 제스처: ${custom.join(", ")}`
    : "학습된 내 제스처가 없어 기본 제스처만 사용해요. (gesture_studio.py 로 학습 후 serve_web.py 재실행)";
}

async function createRecognizer() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm",
  );
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: "../gesture_recognizer.task", delegate },
    runningMode: "VIDEO",
    numHands: 2,
  });
  try {
    return await GestureRecognizer.createFromOptions(vision, options("GPU"));
  } catch {
    return await GestureRecognizer.createFromOptions(vision, options("CPU"));
  }
}

async function main() {
  setStatus("모델 불러오는 중…");
  model = await loadModel();
  fillTargets();
  recognizer = await createRecognizer();

  setStatus("카메라 권한을 허용해 주세요");
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  } catch (e) {
    setStatus(`카메라를 열 수 없어요 (${e.name}). 다른 웹캠 프로그램을 끄고 새로고침하세요.`);
    return;
  }
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  $("#placeholder").hidden = true;
  requestAnimationFrame(frame);
}

// 테스트용 (개발자 도구에서 확인)
window.__fortune = { toFeatures, predictProba, getModel: () => model, reveal };

main().catch((e) => {
  console.error(e);
  setStatus(`오류: ${e.message}`);
});
