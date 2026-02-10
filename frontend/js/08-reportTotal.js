// ====== 환경 설정 ======
const API_BASE = "http://localhost:8000"; // FastAPI 주소

// ====== GT Profile (하드코딩) ======
// NOTE: 여기 GT는 "정상 허용 범위" 예시입니다. (프로젝트 값에 맞게 조정하세요)
const GT_PROFILES = {
  badminton_grip_v1: {
    KF1: {
      thumb_ip: { min: 6.0, max: 18.0 },
      index_mcp: { min: -8.0, max: -2.0 },
      wrist_flex: { min: 4.0, max: 12.0 },
    },
    KF2: {
      thumb_ip: { min: 8.0, max: 22.5 },
      index_mcp: { min: -8.0, max: -2.0 },
      wrist_flex: { min: 5.0, max: 14.0 },
    },
    KF3: {
      thumb_ip: { min: 5.0, max: 17.0 },
      index_mcp: { min: -9.0, max: -3.0 },
      wrist_flex: { min: 5.0, max: 13.0 },
    },
  },
};

const ACTIVE_GT_PROFILE = "badminton_grip_v1";

function getGtRange(frameName, joint) {
  const profile = GT_PROFILES[ACTIVE_GT_PROFILE];
  const f = profile?.[frameName];
  const r = f?.[joint];
  return r ? { min: Number(r.min), max: Number(r.max) } : null;
}

function inRange(v, min, max) {
  const n = Number(v);
  if (!Number.isFinite(n)) return false;
  return n >= min && n <= max;
}

// ====== 데모: DB에서 가져왔다고 가정하는 JSON ======
function getMockDBPayload() {
  return {
    series: [
      { t: 0.0, thumb_ip: 10.2, index_mcp: -4.8, wrist_flex: 7.5 },
      { t: 0.03, thumb_ip: 50.0, index_mcp: -5.1, wrist_flex: 7.9 },
      { t: 0.06, thumb_ip: -12.3, index_mcp: -5.2, wrist_flex: 8.1 },
      { t: 0.09, thumb_ip: 8.5, index_mcp: -6.0, wrist_flex: 8.4 },
    ],
    events: [
      { t: 0.0, name: "KF1" },
      { t: 0.06, name: "KF2" },
      { t: 0.12, name: "KF3" },
    ],
    meta: {
      sport: "badminton",
      action: "grip_check",
    },
    lang: "ko",
  };
}

// ====== 유틸: 심각도(룰 기반) ======
function severityOf(angle) {
  const v = Math.abs(Number(angle));
  if (!Number.isFinite(v)) return "low";
  if (v >= 12) return "high";
  if (v >= 6) return "medium";
  return "low";
}

// ====== Score (0~100) 계산/렌더링 ======
const SCORE_CIRCUMFERENCE = 289.0; // 2πr where r=46 (matches CSS dasharray)

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

// 임시 점수 규칙: 평균 |오차|가 작을수록 점수↑
// meanAbs=0 => 100점, meanAbs=20deg 이상 => 0점 근처
function computeScoreFromAngles(angles) {
  const nums = Object.values(angles)
    .map((v) => Math.abs(Number(v)))
    .filter(Number.isFinite);

  if (nums.length === 0) return 0;
  const meanAbs = nums.reduce((a, b) => a + b, 0) / nums.length;
  const score = 100 - (meanAbs / 20) * 100;
  return Math.round(clamp(score, 0, 100));
}

function setScore(score) { 

  const s = clamp(Math.round(Number(score) || 0), 0, 100); // 0~100 사이 정수로 클램프

  const numEl = document.getElementById("scoreValue");
  if (numEl) numEl.textContent = String(s);

  const fg = document.getElementById("scoreRingFg");
  if (fg) {
    const dashOffset = SCORE_CIRCUMFERENCE * (1 - s / 100);
    fg.style.strokeDashoffset = String(dashOffset);
  }

  const ring = document.querySelector(".scoreRing");
  if (ring) ring.setAttribute("aria-label", `점수 ${s}점`);
}

function setSummaryText({ headline, sub } = {}) {
  const h = document.getElementById("summaryHeadline");
  const s = document.getElementById("summarySub");
  if (h && typeof headline === "string" && headline.trim()) h.innerHTML = headline;
  if (s && typeof sub === "string" && sub.trim()) s.textContent = sub;
}

// LLM 결과에서 점수/요약을 최대한 유연하게 뽑아오기
function extractScoreAndSummary(reportObj) {
  if (!reportObj || typeof reportObj !== "object") return {};

  const score =
    reportObj.score ??
    reportObj.total_score ??
    reportObj.overall_score ??
    reportObj.final_score ??
    reportObj.result_score ??
    null;

  const oneLine =
    reportObj.one_line ??
    reportObj.oneLine ??
    reportObj.summary ??
    reportObj.overall_summary ??
    null;

  const headline = reportObj.headline ?? reportObj.title ?? null;

  return {
    score: score == null ? null : Number(score),
    sub: oneLine == null ? null : String(oneLine),
    headline: headline == null ? null : String(headline),
  };
}

// ====== 시계열 유틸 ======
function safeNumber(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

// series에서 특정 시각 t에 가장 가까운 샘플(스냅샷)을 고릅니다.
function pickNearestSample(series, t) {
  if (!Array.isArray(series) || series.length === 0) return null;
  const target = safeNumber(t);
  if (target == null) return series[series.length - 1];

  let best = series[0];
  let bestDist = Math.abs((safeNumber(series[0].t) ?? 0) - target);

  for (const s of series) {
    const st = safeNumber(s.t);
    if (st == null) continue;
    const d = Math.abs(st - target);
    if (d < bestDist) {
      best = s;
      bestDist = d;
    }
  }
  return best;
}

// series에서 마지막 샘플을 스냅샷으로 사용
function pickLastSample(series) {
  if (!Array.isArray(series) || series.length === 0) return null;
  return series[series.length - 1];
}

// events 중 마지막 이벤트(KF2 등)를 선택
function pickLastEvent(events) {
  if (!Array.isArray(events) || events.length === 0) return null;
  return events[events.length - 1];
}

// 시계열 샘플(한 프레임)을 angles 객체 형태로 변환
function sampleToAngles(sample) {
  if (!sample || typeof sample !== "object") return {};
  const angles = { ...sample };
  delete angles.t; // LLM에 입력은 “관절 각도 값”이지, 시간값이 아니기 때문에 제거 
  return angles;
}

// ====== DOM 렌더링 ======
function renderMeta(meta, events = [], currentEvent = null) {
  const el = document.getElementById("meta");
  if (!el) return;

  const evText =
    Array.isArray(events) && events.length
      ? events.map((e) => `${e.name}@${Number(e.t).toFixed(2)}s`).join(" · ")
      : "-";

  const curText = currentEvent
    ? `${currentEvent.name} @ ${Number(currentEvent.t).toFixed(2)}s`
    : "-";

  el.innerHTML = `
    <div><b>sport</b>: ${meta?.sport ?? "-"}</div>
    <div><b>action</b>: ${meta?.action ?? "-"}</div>
    <div><b>current</b>: ${curText}</div>
    <div><b>events</b>: ${evText}</div>
  `;
}

function renderTable(angles) {
  const tbody = document.querySelector("#anglesTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  Object.entries(angles).forEach(([joint, deg]) => {
    const num = Number(deg);
    const sev = severityOf(num);

    const gt = getGtRange(window.__CURRENT_FRAME__ || "KF2", joint);
    const pass = gt ? inRange(num, gt.min, gt.max) : null;
    const sevText = pass === null ? sev : `${sev} (${pass ? "PASS" : "FAIL"})`;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${joint}</td>
      <td>${Number.isFinite(num) ? num.toFixed(2) : "-"}</td>
      <td>${sevText}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ====== Chart.js 렌더링 ======
const charts = { series: null, donut: null, radar: null };

function destroyChart(key) {
  if (charts[key]) {
    charts[key].destroy();
    charts[key] = null;
  }
}

// ====== 시계열 차트(라인) + 이벤트 마커 ======
const eventMarkerPlugin = {
  id: "eventMarkerPlugin",
  afterDatasetsDraw(chart, args, pluginOptions) {
    const events = pluginOptions?.events || [];
    if (!Array.isArray(events) || events.length === 0) return;

    const { ctx, chartArea, scales } = chart;
    const xScale = scales.x;
    if (!xScale) return;

    ctx.save();
    ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial";
    ctx.fillStyle = "rgba(17,24,39,.75)";
    ctx.strokeStyle = "rgba(32,201,151,.75)";
    ctx.lineWidth = 1;

    for (const ev of events) {
      const t = safeNumber(ev.t);
      if (t == null) continue;

      const x = xScale.getPixelForValue(t);
      if (x < chartArea.left || x > chartArea.right) continue;

      // vertical line
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();

      // label
      const label = String(ev.name ?? "");
      if (label) {
        ctx.fillText(label, x + 2, chartArea.top + 10);
      }
    }
    ctx.restore();
  },
};

function renderThumbIpSeriesChart(series, events) {
  const ctx = document.getElementById("thumbIpSeriesChart");
  if (!ctx) return;
  destroyChart("series");

  const xs = Array.isArray(series) ? series : [];
  const data = xs
    .map((s) => {
      const t = safeNumber(s.t); // ✅ series의 t 사용
      const y = safeNumber(s.thumb_ip);
      if (t == null || y == null) return null;
      return { x: t, y };
    })
    .filter(Boolean);

  charts.series = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "thumb_ip (deg)",
          data,
          pointRadius: 2,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) =>
              ` ${c.parsed.y.toFixed(2)} deg @ ${c.parsed.x.toFixed(2)}s`,
          },
        },
        eventMarkerPlugin: { events },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "time (s)" },
          ticks: { font: { size: 10 } },
        },
        y: {
          title: { display: true, text: "deg" },
          ticks: { font: { size: 10 } },
        },
      },
    },
    plugins: [eventMarkerPlugin],
  });
}

function renderSeverityDonutChart(angles) {
  const counts = { low: 0, medium: 0, high: 0 };
  Object.values(angles).forEach((deg) => (counts[severityOf(deg)] += 1));

  const ctx = document.getElementById("severityDonutChart");
  if (!ctx) return;
  destroyChart("donut");

  charts.donut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["low", "medium", "high"],
      datasets: [
        { label: "Count", data: [counts.low, counts.medium, counts.high] },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "65%",
      layout: { padding: 0 },
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 10, font: { size: 10 } },
        },
      },
    },
  });
}

function renderAnglesRadarChart(angles) {
  const labels = Object.keys(angles);
  const values = Object.values(angles).map((v) => Math.abs(Number(v)));

  const ctx = document.getElementById("anglesRadarChart");
  if (!ctx) return;
  destroyChart("radar");

  charts.radar = new Chart(ctx, {
    type: "radar",
    data: { labels, datasets: [{ label: "|Error| (deg)", data: values }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: 0 },
      plugins: { legend: { display: false } },
      scales: {
        r: {
          beginAtZero: true,
          pointLabels: { font: { size: 10 } },
          ticks: { font: { size: 9 }, backdropPadding: 0 },
        },
      },
    },
  });
}

// ====== LLM 리포트 생성 호출 ======
async function generateLLMReport(payload) {
  const res = await fetch(`${API_BASE}/api/report/posture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`LLM report failed: ${res.status} ${text}`);
  }

  const data = await res.json(); // { report: {...} }
  return data.report;
}

function renderLLMReport(reportObj) {
  // raw JSON
  const rawEl = document.getElementById("llmReport");
  if (rawEl) rawEl.textContent = JSON.stringify(reportObj, null, 2);

  // top meta
  const createdAtEl = document.getElementById("llmCreatedAt");
  const modelEl = document.getElementById("llmModel");
  if (createdAtEl) createdAtEl.textContent = reportObj?.created_at ?? "-";
  if (modelEl) modelEl.textContent = reportObj?.model ?? "-";

  // severity badge
  const sev = String(reportObj?.overall_severity ?? "low").toLowerCase();
  const badge = document.getElementById("overallSeverity");
  if (badge) {
    badge.classList.remove("severityBadge--low", "severityBadge--medium", "severityBadge--high");
    if (sev === "high") badge.classList.add("severityBadge--high");
    else if (sev === "medium") badge.classList.add("severityBadge--medium");
    else badge.classList.add("severityBadge--low");
    badge.textContent = sev.toUpperCase();
  }

  // summary -> 상단 요약에도 반영
  const summaryText = reportObj?.summary ? String(reportObj.summary) : "-";
  const sumEl = document.getElementById("llmSummary");
  if (sumEl) sumEl.textContent = summaryText;

  // 상단 헤드라인/서브
  const headline =
    sev === "high" ? "주의가 필요합니다" :
    sev === "medium" ? "개선 여지가 있습니다" :
    "잘 하고 있습니다";
  setSummaryText({ headline: `${headline}<br/>(${sev.toUpperCase()})`, sub: summaryText });

  // issues
  const issuesEl = document.getElementById("llmIssues");
  if (issuesEl) {
    const issues = Array.isArray(reportObj?.top_issues) ? reportObj.top_issues : [];
    if (issues.length === 0) {
      issuesEl.innerHTML = `<div class="reportSection__body">-</div>`;
    } else {
      issuesEl.innerHTML = issues
        .map((it) => {
          const joint = it?.joint ?? "-";
          const deg = it?.error_deg ?? "-";
          const interp = it?.interpretation ?? "";
          const why = it?.why_it_matters ?? "";
          const fixArr = Array.isArray(it?.fix) ? it.fix : [];
          const chips = fixArr.map((f) => `<span class="chip">${String(f)}</span>`).join("");
          return `
            <div class="issueItem">
              <div class="issueItem__top">
                <div class="issueItem__joint">${String(joint)}</div>
                <div class="issueItem__deg">${String(deg)}°</div>
              </div>
              ${interp ? `<div class="issueItem__p"><b>해석</b>: ${String(interp)}</div>` : ""}
              ${why ? `<div class="issueItem__p"><b>영향</b>: ${String(why)}</div>` : ""}
              ${chips ? `<div class="issueItem__chips">${chips}</div>` : ""}
            </div>
          `;
        })
        .join("");
    }
  }

  // checklist
  const checklistEl = document.getElementById("llmChecklist");
  if (checklistEl) {
    const items = Array.isArray(reportObj?.quick_checklist) ? reportObj.quick_checklist : [];
    checklistEl.innerHTML = items.map((x) => `<li>${String(x)}</li>`).join("");
  }

  // notes
  const notesEl = document.getElementById("llmNotes");
  if (notesEl) notesEl.textContent = reportObj?.notes ? String(reportObj.notes) : "-";
}

// ====== 초기 로드: DB에서 JSON 가져왔다고 가정 ======
async function loadFromDB() {
  // 실제로 DB 조회 API가 있다면 아래처럼 변경
  // const res = await fetch(`${API_BASE}/api/sessions/latest`);
  // return await res.json();
  return getMockDBPayload();
}

// ====== 부트스트랩 ======
(async function init() {
  const payload = await loadFromDB();

  const series = payload?.series || [];
  const events = payload?.events || [];
  const meta = payload?.meta || {};

  // 현재 프레임: 마지막 이벤트(KF2 등)
  const currentEvent = pickLastEvent(events);
  window.__CURRENT_FRAME__ = currentEvent?.name ?? "KF2";
  const snap = currentEvent
    ? pickNearestSample(series, currentEvent.t)
    : pickLastSample(series);

  // ✅ 스냅샷 -> angles (t 제거)
  const angles = sampleToAngles(snap);

  // 메타/시계열/스냅샷 렌더
  renderMeta(meta, events, currentEvent);
  renderThumbIpSeriesChart(series, events);

  renderTable(angles);
  renderSeverityDonutChart(angles);
  renderAnglesRadarChart(angles);

  // 초기 점수: 스냅샷 angles 기반
  setScore(computeScoreFromAngles(angles));

  // ===== LLM 버튼 =====
  // payload 변수명은 그대로 쓰되, 전송할 때 angles를 "추가"해서 서버 스키마(angles 필수)를 만족시킵니다.
  const btn = document.getElementById("btnGenerate");
  if (btn) {
    btn.addEventListener("click", async () => {
      try {
        const requestPayload = {
          ...payload, // series/events/meta/lang 유지 (서버가 무시해도 무해)
          gt_profile: ACTIVE_GT_PROFILE, // ✅ GT 프로필 추가 -> 서버가 참고 할수 있게함
          angles, // ✅ 필수
          meta: {
            ...(payload?.meta || {}),
            frame: currentEvent?.name ?? "KF2",
            t: safeNumber(currentEvent?.t) ?? safeNumber(snap?.t) ?? null,
          },
        };

        const report = await generateLLMReport(requestPayload);
        renderLLMReport(report);

        // LLM이 점수/요약을 내려주면 UI에 반영 (없으면 기존 점수 유지)
        const extracted = extractScoreAndSummary(report);
        if (Number.isFinite(extracted.score)) setScore(extracted.score);
        if (extracted.sub || extracted.headline) setSummaryText(extracted);
      } catch (e) {
        alert(e.message);
      }
    });
  }
})();