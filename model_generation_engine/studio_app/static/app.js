/* 모델생성 스튜디오 v3.0 — Standalone Client & Interactive 3D Previz Pose Engine */
(() => {
  const state = {
    csrf: "",
    turnaroundViews: [],
    posePresets: [],
    subjects: [],
    selectedSubjectId: null,
    assets: [],
    selectedItemIds: new Set(),
    selectedLocationIds: new Set(),
    activeItemCategory: "all",
    faceDirectionMode: "auto",
    batches: [],
    pose: {
      preset_id: "standing_editorial",
      head_azimuth_deg: 0,
      head_pitch_deg: 0,
      head_roll_deg: 0,
      gaze_mode: "auto",
      torso_azimuth_deg: 0,
      camera_elevation_deg: 0,
      focal_length_mm: 85,
      subject_x_norm: 0.58,
      shot_scale: "three-quarter portrait shot",
      joints_override: {},
    },
    drag: null,
    projectedJoints: {},
    pollTimer: null,
  };

  const BONE_EDGES = [
    ["pelvis", "neck", "#dcdce2"],
    ["neck", "head", "#f4f4fa"],
    ["neck", "l_shoulder", "#5ab4ff"],
    ["l_shoulder", "l_elbow", "#4496f5"],
    ["l_elbow", "l_wrist", "#327de8"],
    ["neck", "r_shoulder", "#ffa550"],
    ["r_shoulder", "r_elbow", "#f58741"],
    ["r_elbow", "r_wrist", "#eb6e32"],
    ["pelvis", "l_hip", "#6edc96"],
    ["l_hip", "l_knee", "#55c382"],
    ["l_knee", "l_ankle", "#41aa6e"],
    ["pelvis", "r_hip", "#d778f0"],
    ["r_hip", "r_knee", "#be5fd8"],
    ["r_knee", "r_ankle", "#a54bc3"],
  ];

  const CATEGORY_UI_SCHEMA = {
    garment: {
      subRoles: [
        ["outerwear", "겉옷 / 아우터 (자켓·코트·점퍼)"],
        ["inner_top", "속옷 / 이너 상의 (티셔츠·셔츠·터틀넥)"],
        ["mid_layer", "미들 레이어 (니트·베스트·가디건)"],
        ["bottom", "하의 (슬랙스·데님·스커트)"],
        ["one_piece", "한벌 (수트·원피스·셋업)"],
      ],
      modes: [
        ["auto", "착용방식: 자동 레이어링 (속옷 보이게 자동 오픈)"],
        ["open_layered", "앞여밈 오픈 (속 티셔츠/셔츠 노출)"],
        ["closed_buttoned", "단추/지퍼 깔끔히 잠금"],
        ["sleeves_rolled", "소매 자연스럽게 걷기"],
        ["tucked_in", "상의를 하의 안에 넣어입기 (Tuck-in)"],
        ["untucked_relaxed", "상의를 하의 밖으로 빼입기"],
      ],
    },
    accessory: {
      subRoles: [
        ["watch_left", "손목시계 — 왼손목 (40mm 비례잠금·소매노출)"],
        ["watch_right", "손목시계 — 오른손목 (40mm 비례잠금·소매노출)"],
        ["eyewear", "안경 / 선글라스 (두상 폭 1:1·안경다리 연속)"],
        ["necklace_tie_scarf", "넥타이 / 목걸이 / 스카프 (V존 비례)"],
        ["jewelry_fine", "귀걸이 / 반지 / 팔찌 (1~2.5cm 미세 주얼리)"],
        ["belt", "벨트 (3.2cm 버클 허리선 비례)"],
        ["hat", "모자 / 헤드웨어 (두상 밀착)"],
      ],
      modes: [
        ["auto", "스케일: 인체 대비 표준 실측 비율 잠금"],
        ["delicate", "슬림 / 미니멀 스케일"],
        ["bold", "볼드 / 포인트 스케일 (과장 금지)"],
      ],
    },
    bag: {
      subRoles: [
        ["briefcase_tote", "토트백 / 브리프케이스 / 탑핸들백"],
        ["shoulder_bag", "숄더백 / 호보백"],
        ["crossbody_bag", "크로스백 / 메신저백"],
        ["clutch_pouch", "클러치 / 파우치"],
        ["backpack", "백팩"],
      ],
      modes: [
        ["hand_right", "위치: 오른손으로 탑핸들 자연스럽게 들기"],
        ["hand_left", "위치: 왼손으로 탑핸들 자연스럽게 들기"],
        ["crook_right", "위치: 오른팔 팔꿈치 안쪽에 걸치기"],
        ["crook_left", "위치: 왼팔 팔꿈치 안쪽에 걸치기"],
        ["shoulder_right", "위치: 오른쪽 어깨에 숄더로 메기"],
        ["shoulder_left", "위치: 왼쪽 어깨에 숄더로 메기"],
        ["crossbody_l2r", "위치: 크로스백 (왼어깨 → 오른골반)"],
        ["crossbody_r2l", "위치: 크로스백 (오른어깨 → 왼골반)"],
        ["underarm_clutch", "위치: 옆구리/겨드랑이에 클러치로 끼기"],
        ["backpack", "위치: 양쪽 어깨에 백팩 메기"],
        ["placed_beside", "위치: 들지 않고 옆 바닥/테이블에 내려놓기"],
      ],
    },
    shoe: {
      subRoles: [
        ["dress_shoes", "드레스 구두 / 더비 / 옥스퍼드"],
        ["loafers", "로퍼 (페니·태슬)"],
        ["sneakers", "미니멀 스니커즈"],
        ["boots", "앵클 / 첼시 부츠"],
        ["heels_sandals", "힐 / 스트랩 샌들"],
      ],
      modes: [["auto", "스케일: 전신 풀바디 발 비율 자동 연동"]],
    },
  };

  function getActiveSelectedBag() {
    for (const a of state.assets) {
      if (a.category === "bag" && state.selectedItemIds.has(a.id)) {
        return a;
      }
    }
    return null;
  }

  function populateCategoryControls() {
    const effCat = state.activeItemCategory === "all" ? "garment" : state.activeItemCategory;
    const schema = CATEGORY_UI_SCHEMA[effCat] || CATEGORY_UI_SCHEMA.garment;
    const subSel = $("selItemSubRole");
    const modeSel = $("selItemMode");
    const bagSzSel = $("selBagSize");
    if (!subSel || !modeSel) return;

    subSel.innerHTML = "";
    for (const [val, lbl] of schema.subRoles) {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = lbl;
      subSel.appendChild(opt);
    }
    modeSel.innerHTML = "";
    for (const [val, lbl] of schema.modes) {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = lbl;
      modeSel.appendChild(opt);
    }
    if (bagSzSel) {
      bagSzSel.classList.toggle("hidden", effCat !== "bag");
    }
  }

  function wireDropTarget(el, onFiles) {
    if (!el) return;
    ["dragenter", "dragover"].forEach((evName) => {
      el.addEventListener(evName, (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        el.classList.add("drag-over");
      });
    });
    ["dragleave", "dragend"].forEach((evName) => {
      el.addEventListener(evName, (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        el.classList.remove("drag-over");
      });
    });
    el.addEventListener("drop", async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      el.classList.remove("drag-over");
      const files = ev.dataTransfer && ev.dataTransfer.files ? Array.from(ev.dataTransfer.files) : [];
      const imgFiles = files.filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png|webp|heic)$/i.test(f.name));
      if (imgFiles.length) {
        await onFiles(imgFiles);
      }
    });
  }

  function setFaceDirectionMode(mode) {
    state.faceDirectionMode = mode;
    const btnAuto = $("btnAutoFaceDir");
    if (btnAuto) {
      btnAuto.classList.toggle("hidden", mode === "auto");
    }
    drawPrevizCanvas();
  }

  const $ = (id) => document.getElementById(id);

  function showToast(msg) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.remove("hidden");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => el.classList.add("hidden"), 3400);
  }

  async function apiFetch(url, opts = {}) {
    const headers = Object.assign({}, opts.headers || {});
    if (state.csrf && opts.method && opts.method !== "GET") {
      headers["x-studio-csrf"] = state.csrf;
    }
    const resp = await fetch(url, Object.assign({}, opts, { headers }));
    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const err = await resp.json();
        detail = err.detail || JSON.stringify(err);
      } catch (_) {}
      throw new Error(detail);
    }
    return resp.json();
  }

  function formatDeg(val) {
    const n = Number(val);
    const s = Number.isInteger(n) ? String(n) : n.toFixed(1);
    return (n > 0 ? "+" : "") + s + "°";
  }

  function resolveClosest8Angle(azDeg) {
    const clamped = Math.max(-90, Math.min(90, Number(azDeg) || 0));
    let best = { view: "front", label: "기준 원본 (0°)", azimuth_deg: 0 };
    let bestDist = 999;
    for (const v of state.turnaroundViews) {
      const diff = Math.abs(clamped - v.azimuth_deg);
      if (diff < bestDist) {
        bestDist = diff;
        best = v;
      }
    }
    return best;
  }

  // ── 3D Math & Interactive Canvas Renderer ──
  function rotY(pt, deg) {
    const r = (deg * Math.PI) / 180;
    const c = Math.cos(r);
    const s = Math.sin(r);
    // Negative deg (-90..0) turns toward image-left (-X), positive deg (0..+90) turns toward image-right (+X)
    return [c * pt[0] + s * pt[2], pt[1], -s * pt[0] + c * pt[2]];
  }

  function rotX(pt, deg) {
    const r = (deg * Math.PI) / 180;
    const c = Math.cos(r);
    const s = Math.sin(r);
    return [pt[0], c * pt[1] - s * pt[2], s * pt[1] + c * pt[2]];
  }

  function getActiveRawJoints() {
    const preset =
      state.posePresets.find((p) => p.id === state.pose.preset_id) || state.posePresets[0];
    const base = {};
    if (preset && preset.joints) {
      for (const [k, v] of Object.entries(preset.joints)) {
        base[k] = [v[0], v[1], v[2]];
      }
    }
    const activeBag = getActiveSelectedBag();
    if (activeBag) {
      const cm = activeBag.carry_mode || "hand_right";
      if (cm === "hand_right") {
        base.r_elbow = [0.25, 0.23, 0.04];
        base.r_wrist = [0.25, -0.05, 0.06];
      } else if (cm === "hand_left") {
        base.l_elbow = [-0.25, 0.23, 0.04];
        base.l_wrist = [-0.25, -0.05, 0.06];
      } else if (cm === "crook_right") {
        base.r_elbow = [0.25, 0.2, 0.06];
        base.r_wrist = [0.14, 0.24, 0.2];
      } else if (cm === "crook_left") {
        base.l_elbow = [-0.25, 0.2, 0.06];
        base.l_wrist = [-0.14, 0.24, 0.2];
      }
    }
    for (const [k, v] of Object.entries(state.pose.joints_override || {})) {
      base[k] = [v[0], v[1], v[2]];
    }
    return base;
  }

  function drawPrevizCanvas() {
    const canvas = $("previzCanvas");
    if (!canvas || !state.posePresets.length) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = "#181c24";
    ctx.fillRect(0, 0, w, h);

    // Subtle rule-of-thirds composition lines
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    for (const frac of [1 / 3, 2 / 3]) {
      ctx.beginPath();
      ctx.moveTo(w * frac, 0);
      ctx.lineTo(w * frac, h);
      ctx.stroke();
    }

    const raw = getActiveRawJoints();
    const neckBase = raw.neck || [0, 0.56, 0];
    const world = {};
    for (const [name, xyz] of Object.entries(raw)) {
      if (name === "head") {
        let rel = [xyz[0] - neckBase[0], xyz[1] - neckBase[1], xyz[2] - neckBase[2]];
        rel = rotX(rel, -state.pose.head_pitch_deg);
        rel = rotY(rel, state.pose.head_azimuth_deg);
        const nbRot = rotY(neckBase, state.pose.torso_azimuth_deg);
        world[name] = [nbRot[0] + rel[0], nbRot[1] + rel[1], nbRot[2] + rel[2]];
      } else {
        world[name] = rotY(xyz, state.pose.torso_azimuth_deg);
      }
    }

    const scaleStr = state.pose.shot_scale || "";
    const camDist = scaleStr.includes("medium") ? 2.5 : scaleStr.includes("three-quarter") ? 2.9 : 3.6;
    const fovScale = (state.pose.focal_length_mm / 85.0) * (h * 0.78);
    const cxPx = state.pose.subject_x_norm * w;
    const cyPx = h * (scaleStr.includes("full-body") ? 0.52 : 0.58);

    function project(pt3) {
      const shifted = [pt3[0], pt3[1] - 0.12, pt3[2]];
      const pCam = rotX(shifted, state.pose.camera_elevation_deg);
      const z = Math.max(0.4, camDist - pCam[2]);
      return {
        x: cxPx + (pCam[0] * fovScale) / z,
        y: cyPx - (pCam[1] * fovScale) / z,
        z,
      };
    }

    // Perspective floor grid
    ctx.strokeStyle = "#2b313d";
    ctx.lineWidth = 1;
    for (let gx = -1.2; gx <= 1.21; gx += 0.3) {
      const p1 = project([gx, -0.88, -0.8]);
      const p2 = project([gx, -0.88, 0.8]);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
    for (let gz = -0.8; gz <= 0.81; gz += 0.266) {
      const p1 = project([-1.2, -0.88, gz]);
      const p2 = project([1.2, -0.88, gz]);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    const proj = {};
    for (const [k, v] of Object.entries(world)) {
      proj[k] = project(v);
    }
    state.projectedJoints = { proj, fovScale, camDist };

    const sortedBones = [...BONE_EDGES].sort(
      (a, b) => 0.5 * (proj[b[0]].z + proj[b[1]].z) - 0.5 * (proj[a[0]].z + proj[a[1]].z)
    );
    for (const [j1, j2, col] of sortedBones) {
      const p1 = proj[j1];
      const p2 = proj[j2];
      if (!p1 || !p2) continue;
      ctx.strokeStyle = col;
      ctx.lineCap = "round";
      ctx.lineWidth = Math.max(3, 15 / Math.max(1, 0.5 * (p1.z + p2.z)));
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    for (const [name, p] of Object.entries(proj)) {
      if (name === "head") {
        const rHead = Math.max(13, (0.115 * fovScale) / p.z);
        ctx.fillStyle = "#eceff6";
        ctx.beginPath();
        ctx.arc(p.x, p.y, rHead, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#5a96ff";
        ctx.lineWidth = 2;
        ctx.stroke();

        let noseLocal = rotX([0, 0, 0.15], -state.pose.head_pitch_deg);
        noseLocal = rotY(noseLocal, state.pose.head_azimuth_deg);
        const noseWorld = [
          world.head[0] + noseLocal[0],
          world.head[1] + noseLocal[1],
          world.head[2] + noseLocal[2],
        ];
        const np = project(noseWorld);
        if (np.z <= p.z + 0.03) {
          ctx.strokeStyle = "#2fd6ff";
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(np.x, np.y);
          ctx.stroke();
          ctx.fillStyle = "#2fd6ff";
          ctx.beginPath();
          ctx.arc(np.x, np.y, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      } else {
        const rJ = Math.max(4.5, 11 / Math.max(1, p.z));
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(p.x, p.y, rJ, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#181c24";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    // Live 3D Proportional Bag Prop on Previz Canvas when a Bag is selected
    const activeBag = getActiveSelectedBag();
    if (activeBag) {
      const szMul = {
        micro_mini: 0.52,
        small_compact: 0.74,
        medium_standard: 1.0,
        large_tote_weekender: 1.32,
      }[activeBag.scale_mode || "medium_standard"] || 1.0;
      const bw3 = 0.24 * szMul;
      const bh3 = 0.20 * szMul;
      const cm = activeBag.carry_mode || "hand_right";
      let anchor = null;
      let strapFrom = null;
      if (cm === "hand_right" && world.r_wrist) {
        anchor = [world.r_wrist[0], world.r_wrist[1] - 0.08 - bh3 * 0.5, world.r_wrist[2]];
        strapFrom = world.r_wrist;
      } else if (cm === "hand_left" && world.l_wrist) {
        anchor = [world.l_wrist[0], world.l_wrist[1] - 0.08 - bh3 * 0.5, world.l_wrist[2]];
        strapFrom = world.l_wrist;
      } else if (cm === "crook_right" && world.r_elbow) {
        anchor = [world.r_elbow[0] + 0.02, world.r_elbow[1] - 0.1 - bh3 * 0.5, world.r_elbow[2] + 0.04];
        strapFrom = world.r_elbow;
      } else if (cm === "crook_left" && world.l_elbow) {
        anchor = [world.l_elbow[0] - 0.02, world.l_elbow[1] - 0.1 - bh3 * 0.5, world.l_elbow[2] + 0.04];
        strapFrom = world.l_elbow;
      } else if (cm === "shoulder_right" && world.r_hip) {
        anchor = [world.r_hip[0] + 0.1, world.r_hip[1] + 0.06, world.r_hip[2]];
        strapFrom = world.r_shoulder;
      } else if (cm === "shoulder_left" && world.l_hip) {
        anchor = [world.l_hip[0] - 0.1, world.l_hip[1] + 0.06, world.l_hip[2]];
        strapFrom = world.l_shoulder;
      } else if (cm === "crossbody_l2r" && world.r_hip) {
        anchor = [world.r_hip[0] + 0.1, world.r_hip[1] + 0.02, world.r_hip[2] + 0.04];
        strapFrom = world.l_shoulder;
      } else if (cm === "crossbody_r2l" && world.l_hip) {
        anchor = [world.l_hip[0] - 0.1, world.l_hip[1] + 0.02, world.l_hip[2] + 0.04];
        strapFrom = world.r_shoulder;
      } else if (cm === "underarm_clutch" && world.r_elbow) {
        anchor = [world.r_elbow[0] - 0.04, world.r_elbow[1] + 0.04, world.r_elbow[2] + 0.06];
      } else if (cm === "placed_beside") {
        anchor = [0.44, -0.88 + bh3 * 0.5, 0.08];
      }
      if (anchor) {
        const bp = project(anchor);
        const pw = Math.max(14, (bw3 * fovScale) / bp.z);
        const ph = Math.max(12, (bh3 * fovScale) / bp.z);
        if (strapFrom) {
          const sp = project(strapFrom);
          ctx.strokeStyle = "#f5b342";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(sp.x, sp.y);
          ctx.lineTo(bp.x, bp.y - ph / 2);
          ctx.stroke();
        }
        ctx.fillStyle = "rgba(245, 179, 66, 0.22)";
        ctx.strokeStyle = "#f5b342";
        ctx.lineWidth = 2;
        ctx.fillRect(bp.x - pw / 2, bp.y - ph / 2, pw, ph);
        ctx.strokeRect(bp.x - pw / 2, bp.y - ph / 2, pw, ph);
      }
    }

    const matched = resolveClosest8Angle(state.pose.head_azimuth_deg);
    const badge = $("matchedAngleBadge");
    if (state.faceDirectionMode === "auto") {
      badge.className = "pill pill-info";
      badge.textContent = "얼굴 방향: 자동 (장소 구도·조명 맞춤)";
      document.querySelectorAll(".angle-slot").forEach((el) => {
        el.classList.remove("matched-active");
      });
    } else {
      badge.className = "pill pill-match";
      badge.textContent = "수동 고정: " + matched.label;
      document.querySelectorAll(".angle-slot").forEach((el) => {
        el.classList.toggle("matched-active", el.dataset.view === matched.view);
      });
    }
  }

  function setupCanvasInteractions() {
    const canvas = $("previzCanvas");
    function getCanvasPos(ev) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: ((ev.clientX - rect.left) / rect.width) * canvas.width,
        y: ((ev.clientY - rect.top) / rect.height) * canvas.height,
      };
    }

    canvas.addEventListener("pointerdown", (ev) => {
      const pt = getCanvasPos(ev);
      const proj = (state.projectedJoints && state.projectedJoints.proj) || {};
      let hitJoint = null;
      let bestD = 18;
      for (const [name, jp] of Object.entries(proj)) {
        const d = Math.hypot(pt.x - jp.x, pt.y - jp.y);
        if (d < bestD) {
          bestD = d;
          hitJoint = name;
        }
      }
      const raw = getActiveRawJoints();
      state.drag = {
        mode: hitJoint ? "joint" : "orbit",
        joint: hitJoint,
        startX: pt.x,
        startY: pt.y,
        origJoint: hitJoint ? [...raw[hitJoint]] : null,
        origTorso: state.pose.torso_azimuth_deg,
        origHead: state.pose.head_azimuth_deg,
      };
      canvas.setPointerCapture(ev.pointerId);
    });

    canvas.addEventListener("pointermove", (ev) => {
      if (!state.drag) return;
      const pt = getCanvasPos(ev);
      const dx = pt.x - state.drag.startX;
      const dy = pt.y - state.drag.startY;
      if (state.drag.mode === "orbit") {
        const deltaDeg = Math.round(dx * 0.45 * 2) / 2;
        state.pose.torso_azimuth_deg = Math.max(-90, Math.min(90, state.drag.origTorso + deltaDeg));
        state.pose.head_azimuth_deg = Math.max(-90, Math.min(90, state.drag.origHead + deltaDeg));
        state.faceDirectionMode = "manual";
        if ($("btnAutoFaceDir")) $("btnAutoFaceDir").classList.remove("hidden");
        syncPoseControlsFromState();
      } else if (state.drag.joint) {
        const { fovScale, camDist } = state.projectedJoints;
        const scale = camDist / Math.max(80, fovScale);
        const worldDelta = rotY([dx * scale, -dy * scale, 0], -state.pose.torso_azimuth_deg);
        const oj = state.drag.origJoint;
        state.pose.joints_override[state.drag.joint] = [
          Number((oj[0] + worldDelta[0]).toFixed(3)),
          Number((oj[1] + worldDelta[1]).toFixed(3)),
          Number((oj[2] + worldDelta[2]).toFixed(3)),
        ];
        drawPrevizCanvas();
      }
    });

    const endDrag = () => {
      state.drag = null;
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
  }

  function syncPoseControlsFromState() {
    $("rngHeadAz").value = state.pose.head_azimuth_deg;
    $("valHeadAz").textContent = formatDeg(state.pose.head_azimuth_deg);
    $("rngTorsoAz").value = state.pose.torso_azimuth_deg;
    $("valTorsoAz").textContent = formatDeg(state.pose.torso_azimuth_deg);
    $("rngHeadPitch").value = Math.round(state.pose.head_pitch_deg);
    $("valHeadPitch").textContent = Math.round(state.pose.head_pitch_deg) + "°";
    if ($("rngHeadRoll")) {
      $("rngHeadRoll").value = Math.round(state.pose.head_roll_deg || 0);
      $("valHeadRoll").textContent = formatDeg(Math.round(state.pose.head_roll_deg || 0));
    }
    if ($("selGazeMode")) {
      $("selGazeMode").value = state.pose.gaze_mode || "auto";
    }
    $("rngCamElev").value = Math.round(state.pose.camera_elevation_deg);
    $("valCamElev").textContent = Math.round(state.pose.camera_elevation_deg) + "°";
    $("rngSubjX").value = Math.round(state.pose.subject_x_norm * 100);
    $("valSubjX").textContent = Math.round(state.pose.subject_x_norm * 100) + "%";
    $("selLens").value = String(state.pose.focal_length_mm);
    document.querySelectorAll(".preset-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.preset === state.pose.preset_id);
    });
    drawPrevizCanvas();
  }

  // ── UI Rendering Functions ──
  function renderSubjects() {
    const chips = $("subjectChips");
    chips.innerHTML = "";
    if (!state.subjects.length) {
      chips.innerHTML = '<span class="mini-label">아직 등록된 인물이 없습니다. 좌측에서 얼굴 사진을 업로드하세요.</span>';
      $("btnGenerate8Angles").disabled = true;
      $("btnOpenMasterSheet").classList.add("hidden");
      renderAngleGrid(null);
      return;
    }

    if (!state.selectedSubjectId || !state.subjects.some((s) => s.id === state.selectedSubjectId)) {
      state.selectedSubjectId = state.subjects[0].id;
    }
    const activeSubj = state.subjects.find((s) => s.id === state.selectedSubjectId);

    for (const s of state.subjects) {
      const btn = document.createElement("div");
      btn.className = "subj-chip" + (s.id === state.selectedSubjectId ? " active" : "");
      const readyGenCnt = Object.entries(s.angles || {}).filter(
        ([k, a]) => k !== "front" && a.status === "ready"
      ).length;
      btn.innerHTML =
        '<img src="' +
        s.canonical_front_url +
        '" alt="" />' +
        "<span>" +
        escapeHtml(s.name) +
        " (" +
        readyGenCnt +
        "/8방향)</span>";
      btn.addEventListener("click", () => {
        state.selectedSubjectId = s.id;
        renderSubjects();
      });
      chips.appendChild(btn);
    }

    $("btnGenerate8Angles").disabled = !activeSubj;
    if (activeSubj && activeSubj.master_sheet_url) {
      $("btnOpenMasterSheet").href = activeSubj.master_sheet_url;
      $("btnOpenMasterSheet").classList.remove("hidden");
    } else {
      $("btnOpenMasterSheet").classList.add("hidden");
    }

    if (activeSubj && activeSubj.blueprint) {
      const bp = activeSubj.blueprint;
      $("blueprintBar").textContent =
        "선택 인물: " +
        activeSubj.name +
        " · 3D 골격 불변 계측 [Eye-Chin: " +
        Number(bp.eye_to_chin_ratio).toFixed(3) +
        " | 얼굴 폭/길이: " +
        Number(bp.width_to_length_ratio).toFixed(3) +
        " | 미간 비율: " +
        Number(bp.iod_to_bbox_width_ratio).toFixed(3) +
        " | 구순 폭 비율: " +
        Number(bp.mouth_to_eye_width_ratio).toFixed(3) +
        "] · 상태: " +
        (activeSubj.turnaround_state || "ready");
    }
    renderAngleGrid(activeSubj);
  }

  function renderAngleGrid(subj) {
    const grid = $("angleGrid");
    grid.innerHTML = "";
    const matched = resolveClosest8Angle(state.pose.head_azimuth_deg);

    for (const v of state.turnaroundViews) {
      const slot = document.createElement("div");
      slot.className =
        "angle-slot" +
        (v.is_origin ? " origin-slot" : "") +
        (v.view === matched.view ? " matched-active" : "");
      slot.dataset.view = v.view;

      const info = subj && subj.angles ? subj.angles[v.view] : null;
      const thumb = document.createElement("div");
      thumb.className = "angle-thumb";
      if (info && info.status === "ready" && info.url) {
        const img = document.createElement("img");
        img.src = info.url;
        img.alt = v.label;
        thumb.appendChild(img);
      } else if (info && info.status === "running") {
        thumb.textContent = "생성 중...";
      } else {
        thumb.textContent = subj ? "미생성 (자동생성 또는 직접첨부)" : v.label;
      }

      thumb.addEventListener("click", () => {
        state.pose.head_azimuth_deg = v.azimuth_deg;
        state.pose.torso_azimuth_deg = v.azimuth_deg;
        setFaceDirectionMode("manual");
        syncPoseControlsFromState();
      });

      if (subj) {
        wireDropTarget(slot, async (imgFiles) => {
          const fd = new FormData();
          fd.append("photo", imgFiles[0]);
          const updated = await apiFetch("/api/subjects/" + subj.id + "/angles/" + v.view, {
            method: "POST",
            body: fd,
          });
          const idx = state.subjects.findIndex((s) => s.id === subj.id);
          if (idx >= 0) state.subjects[idx] = updated;
          renderSubjects();
          showToast(v.label + " 드래그앤드랍 교체 완료!");
        });
      }

      const meta = document.createElement("div");
      meta.className = "angle-meta";
      const lbl = document.createElement("span");
      lbl.textContent = v.label;
      meta.appendChild(lbl);

      if (subj) {
        const upLbl = document.createElement("label");
        upLbl.className = "slot-upload-lbl";
        upLbl.textContent = "교체";
        const inp = document.createElement("input");
        inp.type = "file";
        inp.accept = "image/*";
        inp.hidden = true;
        inp.addEventListener("change", async () => {
          if (!inp.files.length) return;
          const fd = new FormData();
          fd.append("photo", inp.files[0]);
          try {
            const updated = await apiFetch("/api/subjects/" + subj.id + "/angles/" + v.view, {
              method: "POST",
              body: fd,
            });
            const idx = state.subjects.findIndex((s) => s.id === subj.id);
            if (idx >= 0) state.subjects[idx] = updated;
            renderSubjects();
            showToast(v.label + " 각도 이미지가 등록되었습니다.");
          } catch (e) {
            showToast("업로드 오류: " + e.message);
          }
        });
        upLbl.appendChild(inp);
        meta.appendChild(upLbl);
      }

      slot.appendChild(thumb);
      slot.appendChild(meta);
      grid.appendChild(slot);
    }
  }

  function renderAssets() {
    const counts = { garment: 0, accessory: 0, bag: 0, shoe: 0 };
    let totalSel = 0;
    for (const a of state.assets) {
      if (counts[a.category] !== undefined && state.selectedItemIds.has(a.id)) {
        counts[a.category] += 1;
        totalSel += 1;
      }
    }
    if ($("cnt-all")) $("cnt-all").textContent = String(totalSel);
    for (const k of Object.keys(counts)) {
      const el = $("cnt-" + k);
      if (el) el.textContent = String(counts[k]);
    }

    // Styling items for active category (or all)
    const itemGrid = $("stylingAssetGrid");
    itemGrid.innerHTML = "";
    const catItems = state.assets.filter((a) =>
      state.activeItemCategory === "all"
        ? a.category !== "location"
        : a.category === state.activeItemCategory
    );
    if (!catItems.length) {
      itemGrid.innerHTML =
        '<div class="mini-label" style="grid-column:1/-1;padding:18px;">상단 6개 드롭 칸에 옷·악세사리·가방·신발 사진을 툭 떨구면 여기에 표시됩니다.</div>';
    } else {
      for (const item of catItems) {
        itemGrid.appendChild(buildAssetCard(item, state.selectedItemIds));
      }
    }

    // Locations
    const locGrid = $("locationAssetGrid");
    locGrid.innerHTML = "";
    const locItems = state.assets.filter((a) => a.category === "location");
    if (!locItems.length) {
      locGrid.innerHTML =
        '<div class="mini-label" style="grid-column:1/-1;padding:18px;">첨부된 장소/스타일 사진이 없습니다. 여러 장을 올리면 장수만큼 동시에 생성되며, 우측 텍스트만으로도 생성 가능합니다.</div>';
    } else {
      for (const loc of locItems) {
        locGrid.appendChild(buildAssetCard(loc, state.selectedLocationIds));
      }
    }
    updateBatchCountLabel();
    drawPrevizCanvas();
  }

  function buildAssetCard(asset, selectionSet) {
    const selected = selectionSet.has(asset.id);
    const card = document.createElement("div");
    card.className = "asset-card" + (selected ? " selected" : "");

    const img = document.createElement("img");
    img.src = asset.masked_url || asset.preview_url;
    img.alt = asset.filename;
    img.addEventListener("click", () => {
      if (selectionSet.has(asset.id)) selectionSet.delete(asset.id);
      else selectionSet.add(asset.id);
      renderAssets();
    });
    card.appendChild(img);

    const body = document.createElement("div");
    body.className = "asset-card-body";

    const topRow = document.createElement("div");
    topRow.className = "asset-top-row";
    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.checked = selected;
    chk.addEventListener("change", () => {
      if (chk.checked) selectionSet.add(asset.id);
      else selectionSet.delete(asset.id);
      renderAssets();
    });

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "btn-del-mini";
    delBtn.textContent = "✕";
    delBtn.addEventListener("click", async () => {
      await apiFetch("/api/assets/" + asset.id, { method: "DELETE" });
      state.assets = state.assets.filter((x) => x.id !== asset.id);
      selectionSet.delete(asset.id);
      renderAssets();
    });

    topRow.appendChild(chk);
    const titleSpan = document.createElement("span");
    titleSpan.style.flex = "1";
    titleSpan.style.marginLeft = "5px";
    titleSpan.style.overflow = "hidden";
    titleSpan.style.textOverflow = "ellipsis";
    titleSpan.style.whiteSpace = "nowrap";
    titleSpan.textContent = asset.sub_role || asset.filename;
    topRow.appendChild(titleSpan);
    topRow.appendChild(delBtn);
    body.appendChild(topRow);

    if (asset.category === "bag") {
      const carrySel = document.createElement("select");
      carrySel.style.fontSize = "10px";
      carrySel.style.padding = "3px 4px";
      for (const [val, lbl] of CATEGORY_UI_SCHEMA.bag.modes) {
        const o = document.createElement("option");
        o.value = val;
        o.textContent = lbl.replace("위치: ", "");
        if (val === (asset.carry_mode || "hand_right")) o.selected = true;
        carrySel.appendChild(o);
      }
      carrySel.addEventListener("change", async () => {
        asset.carry_mode = carrySel.value;
        await apiFetch("/api/assets/" + asset.id, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ carry_mode: carrySel.value }),
        });
        drawPrevizCanvas();
      });
      body.appendChild(carrySel);
    } else if (asset.category === "location") {
      const locAngleSel = document.createElement("select");
      locAngleSel.style.fontSize = "10px";
      locAngleSel.style.padding = "3px 4px";
      const autoInfo = asset.auto_pose_info;
      const autoTxt = autoInfo ? "자동 (" + autoInfo.auto_matched_label + ")" : "자동 (구도·조명 맞춤)";
      const oAuto = document.createElement("option");
      oAuto.value = "auto";
      oAuto.textContent = autoTxt;
      if (asset.override_yaw_deg === null || asset.override_yaw_deg === undefined) oAuto.selected = true;
      locAngleSel.appendChild(oAuto);
      for (const v of state.turnaroundViews) {
        const o = document.createElement("option");
        o.value = String(v.azimuth_deg);
        o.textContent = "수동 고정: " + v.label;
        if (asset.override_yaw_deg !== null && asset.override_yaw_deg !== undefined && Number(asset.override_yaw_deg) === v.azimuth_deg) {
          o.selected = true;
        }
        locAngleSel.appendChild(o);
      }
      locAngleSel.addEventListener("change", async () => {
        const val = locAngleSel.value === "auto" ? null : Number(locAngleSel.value);
        asset.override_yaw_deg = val;
        await apiFetch("/api/assets/" + asset.id, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ override_yaw_deg: locAngleSel.value }),
        });
      });
      body.appendChild(locAngleSel);
    } else if (asset.category !== "location") {
      const metaTag = document.createElement("div");
      metaTag.className = "mini-label";
      metaTag.style.fontSize = "10px";
      const modeInfo =
        asset.category === "garment"
          ? asset.wearing_style || "auto"
          : asset.size_cm
          ? asset.size_cm + "cm"
          : asset.scale_mode || "비례잠금";
      metaTag.textContent = (asset.sub_role || asset.category) + " · " + modeInfo;
      body.appendChild(metaTag);
    }

    if (asset.foreign_faces_masked > 0) {
      const mTag = document.createElement("div");
      mTag.className = "mask-tag";
      mTag.textContent = "외부 얼굴 " + asset.foreign_faces_masked + "개 마스킹됨";
      body.appendChild(mTag);
    }
    card.appendChild(body);
    return card;
  }

  function getTextLocationLines() {
    return ($("textLocationsInput").value || "")
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  function updateBatchCountLabel() {
    const imgLocCount = state.assets.filter(
      (a) => a.category === "location" && state.selectedLocationIds.has(a.id)
    ).length;
    const txtLocCount = getTextLocationLines().length;
    const total = Math.max(1, imgLocCount + txtLocCount);
    $("btnLaunchBatch").textContent = "총 " + total + "장 동시 생성 시작";
  }

  function renderBatches() {
    const box = $("batchesContainer");
    box.innerHTML = "";
    if (!state.batches.length) {
      box.innerHTML = '<div class="mini-label">아직 실행된 배치가 없습니다. 상단에서 설정 후 동시 생성을 시작하세요.</div>';
      return;
    }

    for (const b of state.batches) {
      const blk = document.createElement("div");
      blk.className = "batch-block";

      const head = document.createElement("div");
      head.className = "batch-head";
      head.innerHTML =
        "<div><strong>" +
        escapeHtml(b.subject_name) +
        "</strong> · " +
        escapeHtml(b.canvas_preset) +
        " (" +
        b.target_size[0] +
        "×" +
        b.target_size[1] +
        ") · 총 " +
        b.total_jobs +
        "장 · 엔진: " +
        escapeHtml(b.provider) +
        '</div><span class="pill pill-outline">' +
        escapeHtml(b.status) +
        "</span>";
      blk.appendChild(head);

      const jGrid = document.createElement("div");
      jGrid.className = "jobs-grid";
      for (const j of b.jobs || []) {
        const card = document.createElement("div");
        card.className = "job-card";

        let stClass = "st-run";
        if (j.status === "done") stClass = "st-done";
        else if (j.status === "manifest_saved") stClass = "st-manifest";
        else if (j.status === "failed") stClass = "st-err";

        card.innerHTML =
          '<div class="asset-top-row"><strong>#' +
          j.job_index +
          " " +
          escapeHtml(j.location_name) +
          '</strong><span class="job-status ' +
          stClass +
          '">' +
          escapeHtml(j.status) +
          '</span></div><div class="mini-label">매칭 8각도: ' +
          escapeHtml(j.matched_8angle_label) +
          " (머리 " +
          Math.round(j.head_azimuth_deg) +
          "°) · 참조 이미지 " +
          (j.reference_images ? j.reference_images.length : 0) +
          "장</div>";

        const media = document.createElement("div");
        media.className = "job-media";
        if (j.hero_url) {
          media.innerHTML =
            '<a href="' +
            j.hero_url +
            '" target="_blank"><img src="' +
            j.hero_url +
            '" alt="5K Hero" /></a>' +
            (j.closeup_url
              ? '<a href="' +
                j.closeup_url +
                '" target="_blank"><img src="' +
                j.closeup_url +
                '" alt="Closeup" /></a>'
              : j.previz_url
              ? '<img src="' + j.previz_url + '" alt="Previz" />'
              : "");
        } else if (j.previz_url) {
          media.innerHTML =
            '<img src="' +
            j.previz_url +
            '" alt="3D Previz Guide" style="grid-column:1/-1;height:165px;object-fit:contain;" />';
        }
        card.appendChild(media);

        if (j.quality) {
          const passed = Boolean(j.quality.passed || j.quality.passed_all_gates);
          const sharp = Number(j.quality.face_laplacian_variance || j.quality.laplacian_variance || 0).toFixed(1);
          const sface =
            j.quality.identity_cosine_sim !== null && j.quality.identity_cosine_sim !== undefined
              ? " · SFace " + Number(j.quality.identity_cosine_sim).toFixed(3)
              : "";
          const qDiv = document.createElement("div");
          qDiv.className = "mini-label";
          qDiv.style.color = passed ? "#2fd6a2" : "#f5b342";
          qDiv.textContent = "9-Gate 검수: " + (passed ? "PASS" : "CHECK") + sface + " · Sharpness " + sharp;
          card.appendChild(qDiv);
        }
        if (j.status === "done") {
          const remRow = document.createElement("div");
          remRow.style.display = "flex";
          remRow.style.gap = "6px";
          remRow.style.alignItems = "center";
          remRow.style.marginTop = "6px";
          const remSel = document.createElement("select");
          remSel.style.fontSize = "11px";
          remSel.style.flex = "1";
          const modes = [
            ["digital_neutral_raw", "필름/컬러: 무보정 클린 디지털 RAW (노란끼 제거)"],
            ["kodak_portra_400", "필름: Kodak Portra 400 (맑은 스킨톤·크리미 뉴트럴)"],
            ["fujifilm_pro_400h", "필름: Fujifilm Pro 400H (청량한 쿨-에어리 데이라이트)"],
            ["cinestill_50d", "필름: CineStill 50D (투명한 시네마 데이라이트)"],
            ["kodak_gold_200", "필름: Kodak Gold 200 (담백한 오후 내추럴 웜)"],
            ["ilford_hp5_bw", "필름: Ilford HP5 Plus 400 (클래식 흑백)"],
            ["off", "후처리 끄기 (생성 원본 그대로)"],
          ];
          const curMode = j.cohesion_mode === "auto" ? "digital_neutral_raw" : j.cohesion_mode || "digital_neutral_raw";
          for (const [mv, ml] of modes) {
            const opt = document.createElement("option");
            opt.value = mv;
            opt.textContent = ml;
            if (curMode === mv) opt.selected = true;
            remSel.appendChild(opt);
          }
          remSel.addEventListener("change", async () => {
            try {
              showToast("무드·톤·명암·색감 즉시 재마스터링 중 (재생성 없음)...");
              await apiFetch("/api/batches/" + b.id + "/jobs/" + j.job_index + "/remaster", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ cohesion_mode: remSel.value }),
              });
              await loadStatus();
              showToast("후처리 광학 재마스터링 적용 완료!");
            } catch (err) {
              showToast("재마스터링 오류: " + err.message);
            }
          });
          remRow.appendChild(remSel);
          card.appendChild(remRow);
        }
        if (j.error) {
          const eDiv = document.createElement("div");
          eDiv.className = "mini-label";
          eDiv.style.color = "#ff5d6c";
          eDiv.textContent = "오류: " + j.error;
          card.appendChild(eDiv);
        }

        const det = document.createElement("details");
        det.innerHTML =
          "<summary>방화벽 검증 프롬프트 &amp; 참조 계보 보기</summary><pre>" +
          escapeHtml(j.prompt) +
          "</pre>";
        card.appendChild(det);

        jGrid.appendChild(card);
      }
      blk.appendChild(jGrid);
      box.appendChild(blk);
    }
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function loadStatus() {
    const data = await apiFetch("/api/status");
    state.csrf = data.csrf;
    state.turnaroundViews = data.turnaround_views || [];
    state.posePresets = data.pose_presets || [];
    state.subjects = data.subjects || [];
    state.assets = data.assets || [];
    state.batches = data.batches || [];

    const rt = data.runtime || {};
    $("runtimePill").textContent =
      "실행 엔진: " +
      (rt.gti_installed || rt.gti_available
        ? "GTI 로컬 준비됨 (병렬 생성)"
        : rt.gemini_configured || rt.gemini_api_available
        ? "Gemini API 준비됨"
        : "프롬프트·프리비즈 번들 모드");

    // Populate pose preset buttons once
    const pRow = $("posePresetButtons");
    if (!pRow.children.length && state.posePresets.length) {
      for (const p of state.posePresets) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "preset-btn" + (p.id === state.pose.preset_id ? " active" : "");
        b.dataset.preset = p.id;
        b.textContent = p.label;
        b.addEventListener("click", () => {
          state.pose.preset_id = p.id;
          state.pose.head_azimuth_deg = p.head_azimuth_deg;
          state.pose.head_pitch_deg = p.head_pitch_deg || 0;
          state.pose.head_roll_deg = p.head_roll_deg || 0;
          state.pose.gaze_mode = p.gaze_mode || "auto";
          state.pose.torso_azimuth_deg = p.torso_azimuth_deg;
          state.pose.shot_scale = p.shot_scale;
          state.pose.joints_override = {};
          setFaceDirectionMode("manual");
          syncPoseControlsFromState();
        });
        pRow.appendChild(b);
      }
    }

    renderSubjects();
    renderAssets();
    renderBatches();
    syncPoseControlsFromState();
    ensurePollingIfActive();
  }

  function ensurePollingIfActive() {
    const anyRunning =
      state.subjects.some((s) => s.turnaround_state === "generating") ||
      state.batches.some((b) => b.status === "running");
    if (anyRunning && !state.pollTimer) {
      state.pollTimer = setInterval(async () => {
        try {
          await loadStatus();
        } catch (_) {}
      }, 3000);
    } else if (!anyRunning && state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  async function registerFaceFile(file) {
    const fd = new FormData();
    fd.append("photo", file);
    const autoName = ($("subjName").value || "").trim() || file.name.replace(/\.[^.]+$/, "");
    fd.append("name", autoName);
    fd.append("identity_label", $("subjLabel").value || "human subject in the reference portrait");
    if ($("subjHeight").value) fd.append("height_cm", $("subjHeight").value);
    if ($("subjWeight").value) fd.append("weight_kg", $("subjWeight").value);
    fd.append("eyewear", $("subjEyewear").value || "");
    fd.append("hairstyle", $("subjHair").value || "");

    $("btnRegisterSubject").disabled = true;
    try {
      const created = await apiFetch("/api/subjects", { method: "POST", body: fd });
      state.subjects.push(created);
      state.selectedSubjectId = created.id;
      $("subjectForm").reset();
      $("subjLabel").value = "human subject in the reference portrait";
      renderSubjects();
      showToast(created.name + " 드래그앤드랍 즉시 등록 완료!");
    } catch (e) {
      showToast("등록 실패: " + e.message);
    } finally {
      $("btnRegisterSubject").disabled = false;
    }
  }

  async function uploadItemsQuick(files, category, subRole, styleOrMode, scaleMode, sizeCm, desc) {
    if (!files.length) return;
    const fd = new FormData();
    fd.append("category", category);
    fd.append("sub_role", subRole || "");
    fd.append("description", desc || "");
    if (category === "garment") {
      fd.append("wearing_style", styleOrMode || "auto");
    } else if (category === "accessory") {
      fd.append("scale_mode", styleOrMode || "auto");
    } else if (category === "bag") {
      fd.append("carry_mode", styleOrMode || "hand_right");
      fd.append("scale_mode", scaleMode || "medium_standard");
    }
    if (sizeCm) fd.append("size_cm", sizeCm);
    for (const f of files) fd.append("files", f);

    try {
      const res = await apiFetch("/api/assets/upload", { method: "POST", body: fd });
      for (const a of res.uploaded) {
        state.assets.push(a);
        if (category === "location") state.selectedLocationIds.add(a.id);
        else state.selectedItemIds.add(a.id);
      }
      renderAssets();
      showToast(res.count + "장 드래그앤드랍 첨부 완료!");
    } catch (e) {
      showToast("첨부 실패: " + e.message);
    }
  }

  function bindEvents() {
    // 1. Face Dropzone: instant upload on drop OR file select
    wireDropTarget($("faceDropzone"), async (files) => {
      await registerFaceFile(files[0]);
    });
    $("faceInput").addEventListener("change", () => {
      if ($("faceInput").files.length) {
        $("faceDropLabel").innerHTML =
          "<strong>선택됨: " + escapeHtml($("faceInput").files[0].name) + "</strong><span>하단 버튼을 눌러 등록하세요</span>";
      }
    });

    $("subjectForm").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      if (!$("faceInput").files.length) {
        showToast("얼굴 원본 사진을 선택하거나 드롭하세요.");
        return;
      }
      await registerFaceFile($("faceInput").files[0]);
    });

    // 2. 6-Slot Quick Drag & Drop Pads in STEP 02
    document.querySelectorAll(".quick-drop-pad").forEach((pad) => {
      const qcat = pad.dataset.qcat;
      const qrole = pad.dataset.qrole;
      const qstyle = pad.dataset.qstyle;
      const inp = pad.querySelector("input[type='file']");
      wireDropTarget(pad, async (files) => {
        await uploadItemsQuick(files, qcat, qrole, qstyle, "medium_standard", "", "");
      });
      if (inp) {
        inp.addEventListener("change", async () => {
          if (inp.files.length) {
            await uploadItemsQuick(Array.from(inp.files), qcat, qrole, qstyle, "medium_standard", "", "");
            inp.value = "";
          }
        });
      }
    });

    // 3. Location Multi-File Dropzone in STEP 04
    wireDropTarget($("locationDropzone"), async (files) => {
      await uploadItemsQuick(files, "location", "", "auto", "", "", "");
    });
    wireDropTarget($("locationAssetGrid"), async (files) => {
      await uploadItemsQuick(files, "location", "", "auto", "", "", "");
    });

    $("btnGenerate8Angles").addEventListener("click", async () => {
      if (!state.selectedSubjectId) return;
      try {
        const provider = $("selProvider").value;
        const res = await apiFetch("/api/subjects/" + state.selectedSubjectId + "/generate-8angles", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider }),
        });
        showToast(
          res.mode === "manifest"
            ? "8방향 프롬프트 설계도가 저장되었습니다."
            : "좌·우 디테일 8방향 자동 생성을 시작했습니다!"
        );
        await loadStatus();
      } catch (e) {
        showToast("8방향 생성 오류: " + e.message);
      }
    });

    document.querySelectorAll(".cat-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".cat-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        state.activeItemCategory = tab.dataset.cat;
        populateCategoryControls();
        renderAssets();
      });
    });

    $("itemFilesInput").addEventListener("change", async () => {
      const files = Array.from($("itemFilesInput").files || []);
      if (!files.length) return;
      const effCat = state.activeItemCategory === "all" ? "garment" : state.activeItemCategory;
      await uploadItemsQuick(
        files,
        effCat,
        $("selItemSubRole").value || "",
        $("selItemMode").value || "auto",
        $("selBagSize").value || "medium_standard",
        $("itemSizeCm").value || "",
        $("itemDesc").value || ""
      );
      $("itemFilesInput").value = "";
    });

    $("locationFilesInput").addEventListener("change", async () => {
      const files = Array.from($("locationFilesInput").files || []);
      if (!files.length) return;
      await uploadItemsQuick(files, "location", "", "auto", "", "", "");
      $("locationFilesInput").value = "";
    });

    $("textLocationsInput").addEventListener("input", updateBatchCountLabel);

    if ($("btnAutoFaceDir")) {
      $("btnAutoFaceDir").addEventListener("click", () => {
        setFaceDirectionMode("auto");
        showToast("얼굴 방향이 자동(장소·구도 맞춤) 모드로 전환되었습니다.");
      });
    }

    // Pose Sliders (moving head/torso switches to manual direction override)
    $("rngHeadAz").addEventListener("input", (e) => {
      state.pose.head_azimuth_deg = Number(e.target.value);
      setFaceDirectionMode("manual");
      syncPoseControlsFromState();
    });
    $("rngTorsoAz").addEventListener("input", (e) => {
      state.pose.torso_azimuth_deg = Number(e.target.value);
      setFaceDirectionMode("manual");
      syncPoseControlsFromState();
    });
    $("rngHeadPitch").addEventListener("input", (e) => {
      state.pose.head_pitch_deg = Number(e.target.value);
      syncPoseControlsFromState();
    });
    if ($("rngHeadRoll")) {
      $("rngHeadRoll").addEventListener("input", (e) => {
        state.pose.head_roll_deg = Number(e.target.value);
        syncPoseControlsFromState();
      });
    }
    if ($("selGazeMode")) {
      $("selGazeMode").addEventListener("change", (e) => {
        state.pose.gaze_mode = e.target.value;
        drawPrevizCanvas();
      });
    }
    $("rngCamElev").addEventListener("input", (e) => {
      state.pose.camera_elevation_deg = Number(e.target.value);
      syncPoseControlsFromState();
    });
    $("rngSubjX").addEventListener("input", (e) => {
      state.pose.subject_x_norm = Number(e.target.value) / 100;
      syncPoseControlsFromState();
    });
    $("selLens").addEventListener("change", (e) => {
      state.pose.focal_length_mm = Number(e.target.value);
      drawPrevizCanvas();
    });
    $("selShotScale").addEventListener("change", (e) => {
      state.pose.shot_scale = e.target.value;
      drawPrevizCanvas();
    });
    $("btnResetJoints").addEventListener("click", () => {
      state.pose.joints_override = {};
      drawPrevizCanvas();
      showToast("프리비즈 관절 위치가 프리셋 기본값으로 초기화되었습니다.");
    });

    $("btnLaunchBatch").addEventListener("click", async () => {
      if (!state.selectedSubjectId) {
        showToast("먼저 STEP 01에서 인물을 선택하거나 얼굴 사진을 드롭하세요.");
        return;
      }
      const textItems = [];
      if ($("txtGarmentNote").value.trim()) {
        textItems.push({ category: "garment", description: $("txtGarmentNote").value.trim() });
      }
      if ($("txtAccessoryNote").value.trim()) {
        textItems.push({ category: "accessory", description: $("txtAccessoryNote").value.trim() });
      }

      const payload = {
        subject_id: state.selectedSubjectId,
        item_asset_ids: Array.from(state.selectedItemIds),
        text_items: textItems,
        location_asset_ids: Array.from(state.selectedLocationIds),
        text_locations: getTextLocationLines(),
        shared_mood_prompt: $("sharedMoodPrompt").value.trim(),
        shared_lighting_note: $("sharedLightingNote").value.trim(),
        location_scope: $("selLocationScope").value,
        canvas_preset: $("selCanvasPreset").value,
        provider: $("selProvider").value,
        camera_id: $("selCameraBody") ? $("selCameraBody").value : "hasselblad_x2d",
        film_id: $("selFilmStock") ? $("selFilmStock").value : "digital_neutral_raw",
        lighting_mode: $("selLightingMode") ? $("selLightingMode").value : "auto",
        aperture_mode: $("selApertureMode") ? $("selApertureMode").value : "auto",
        cohesion_mode: $("selFilmStock") ? $("selFilmStock").value : "digital_neutral_raw",
        face_direction_mode: state.faceDirectionMode,
        attach_previz_image: false,
        pose: Object.assign({}, state.pose, {
          custom_pose_note: $("customPoseNote").value.trim(),
        }),
      };

      $("btnLaunchBatch").disabled = true;
      try {
        const batch = await apiFetch("/api/batches", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        showToast("총 " + batch.total_jobs + "장 1원본들→1생성 배치가 시작되었습니다!");
        await loadStatus();
      } catch (e) {
        showToast("배치 실행 오류: " + e.message);
      } finally {
        $("btnLaunchBatch").disabled = false;
        updateBatchCountLabel();
      }
    });

    $("btnRefreshStatus").addEventListener("click", loadStatus);
    setupCanvasInteractions();
  }

  window.addEventListener("DOMContentLoaded", async () => {
    populateCategoryControls();
    bindEvents();
    try {
      await loadStatus();
    } catch (e) {
      showToast("초기 로드 오류: " + e.message);
    }
  });
})();
