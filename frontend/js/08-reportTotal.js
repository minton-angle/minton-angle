// ====== 환경 설정 ======
const API_BASE = "http://localhost:8000"; // FastAPI 주소

// ====== GT Profile (종합 리포트 페이지에서는 사용 안 함) ======
// NOTE: 종합 페이지는 세션 히스토리 기반이므로, GT 범위는 서버/다른 페이지에서 처리하는 것을 권장합니다.

// ====== 데모: DB에서 가져왔다고 가정하는 JSON(세션 히스토리) ======
function getMockDBPayload() {
  return {
    user_id: 1,
    sessions: [
      {
        idx: 1,
        created_at: "2026-02-01",
        frame: "KF3",
        angles: { thumb_ip: 14.2, index_mcp: -6.8, wrist_flex: 10.5 },
        meta: { sport: "badminton", action: "swing_check" },
        lang: "ko",
      },
      {
        idx: 2,
        created_at: "2026-02-03",
        frame: "KF3",
        angles: { thumb_ip: 11.0, index_mcp: -5.1, wrist_flex: 8.2 },
        meta: { sport: "badminton", action: "swing_check" },
        lang: "ko",
      },
      {
        idx: 3,
        created_at: "2026-02-07",
        frame: "KF3",
        angles: { thumb_ip: 8.6, index_mcp: -4.2, wrist_flex: 6.7 },
        meta: { sport: "badminton", action: "swing_check" },
        lang: "ko",
      },
      {
        idx: 4,
        created_at: "2026-02-10",
        frame: "KF3",
        angles: { thumb_ip: 9.3, index_mcp: -4.9, wrist_flex: 7.4 },
        meta: { sport: "badminton", action: "swing_check" },
        lang: "ko",
      },
    ],
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
  const s = clamp(Math.round(Number(score) || 0), 0, 100);

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

// ====== DOM 렌더링 ======
function renderMeta(meta) {
  const el = document.getElementById("meta");
  if (!el) return;

  el.innerHTML = `
    <div><b>sport</b>: ${meta?.sport ?? "-"}</div>
    <div><b>action</b>: ${meta?.action ?? "-"}</div>
    <div><b>session</b>: ${meta?.idx ?? "-"}</div>
    <div><b>date</b>: ${meta?.created_at ?? "-"}</div>
    <div><b>frame</b>: ${window.__CURRENT_FRAME__ ?? "-"}</div>
  `;
}

function renderTable(angles) {
  const tbody = document.querySelector("#anglesTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  Object.entries(angles || {}).forEach(([joint, deg]) => {
    const num = Number(deg);
    const sev = severityOf(num);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${joint}</td>
      <td>${Number.isFinite(num) ? num.toFixed(2) : "-"}</td>
      <td>${sev}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ====== Chart.js 렌더링 ======
const charts = { scoreHistory: null, donut: null, radar: null };

function destroyChart(key) {
  if (charts[key]) {
    charts[key].destroy();
    charts[key] = null;
  }
}

// ✅ 세션 점수 히스토리 차트
function renderScoreHistoryChart(sessions) {
  const ctx = document.getElementById("scoreHistoryChart");
  if (!ctx) return;
  destroyChart("scoreHistory");

  const xs = Array.isArray(sessions) ? sessions : [];
  const labels = xs.map((s) => s.created_at ?? `#${s.idx ?? "-"}`);
  const values = xs.map((s) => {
    const direct = Number(s.score);
    if (Number.isFinite(direct)) return direct;
    return computeScoreFromAngles(s.angles || {});
  });

  charts.scoreHistory = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{ label: "score", data: values, pointRadius: 2, tension: 0.25 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => ` ${c.parsed.y} 점`,
          },
        },
      },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { min: 0, max: 100, ticks: { font: { size: 10 } } },
      },
    },
  });
}

function renderSeverityDonutChart(angles) {
  const counts = { low: 0, medium: 0, high: 0 };
  Object.values(angles || {}).forEach((deg) => (counts[severityOf(deg)] += 1));

  const ctx = document.getElementById("severityDonutChart");
  if (!ctx) return;
  destroyChart("donut");

  charts.donut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["low", "medium", "high"],
      datasets: [{ label: "Count", data: [counts.low, counts.medium, counts.high] }],
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
  const labels = Object.keys(angles || {});
  const values = Object.values(angles || {}).map((v) => Math.abs(Number(v)));

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

  const data = await res.json(); // { report: { ... } } 또는 report가 최상위
  return data?.report ?? data;
}

function renderLLMReport(reportObj) {
  const DEV = location.hostname === "localhost";
  if (DEV) {
    console.groupCollapsed("[LLM REPORT RAW JSON]");
    console.log(reportObj);
    console.groupEnd();
  }

  const createdAtEl = document.getElementById("llmCreatedAt");
  const modelEl = document.getElementById("llmModel");
  if (createdAtEl) createdAtEl.textContent = reportObj?.created_at ?? "-";
  if (modelEl) modelEl.textContent = reportObj?.model ?? "-";

  const sev = String(reportObj?.overall_severity ?? reportObj?.result ?? "low").toLowerCase();
  const badge = document.getElementById("overallSeverity");
  if (badge) {
    badge.classList.remove("severityBadge--low", "severityBadge--medium", "severityBadge--high");
    if (sev === "high") badge.classList.add("severityBadge--high");
    else if (sev === "medium") badge.classList.add("severityBadge--medium");
    else badge.classList.add("severityBadge--low");
    badge.textContent = sev.toUpperCase();
  }

  const summaryText = reportObj?.summary ? String(reportObj.summary) : "-";
  const sumEl = document.getElementById("llmSummary");
  if (sumEl) sumEl.textContent = summaryText;

  const headline =
    sev === "high" ? "주의가 필요합니다" :
    sev === "medium" ? "개선 여지가 있습니다" :
    "잘 하고 있습니다";
  setSummaryText({ headline: `${headline}<br/>(${sev.toUpperCase()})`, sub: summaryText });

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

  const checklistEl = document.getElementById("llmChecklist");
  if (checklistEl) {
    const items = Array.isArray(reportObj?.quick_checklist) ? reportObj.quick_checklist : [];
    checklistEl.innerHTML = items.map((x) => `<li>${String(x)}</li>`).join("");
  }

  const notesEl = document.getElementById("llmNotes");
  if (notesEl) notesEl.textContent = reportObj?.notes ? String(reportObj.notes) : "-";
}

// ====== 초기 로드: DB에서 JSON 가져왔다고 가정 ======
async function loadFromDB() {
  return getMockDBPayload();
}

// ====== 부트스트랩 ======
(async function init() {
  const payload = await loadFromDB();

  const sessions = Array.isArray(payload?.sessions) ? payload.sessions : [];
  const last = sessions.length ? sessions[sessions.length - 1] : null;

  // 종합 리포트 페이지: 세션 히스토리(여러 번의 평가)를 시각화
  renderScoreHistoryChart(sessions);

  // 현재 세션(가장 최근) 스냅샷
  window.__CURRENT_FRAME__ = last?.frame ?? "KF3";
  const angles = last?.angles || {};
  const meta = last?.meta || {};

  // 메타/스냅샷 렌더
  renderMeta({ ...meta, created_at: last?.created_at, idx: last?.idx });

  renderTable(angles);
  renderSeverityDonutChart(angles);
  renderAnglesRadarChart(angles);

  // 초기 점수: 최근 세션 angles 기반
  setScore(computeScoreFromAngles(angles));

  // ===== LLM 버튼 =====
  const btn = document.getElementById("btnGenerate");
  if (btn) {
    btn.addEventListener("click", async () => {
      try {
        const requestPayload = {
          angles, // ✅ 필수(최근 세션 기준)
          meta: {
            ...(meta || {}),
            frame: window.__CURRENT_FRAME__,
            created_at: last?.created_at ?? null,
            session_idx: last?.idx ?? null,
          },
          history: sessions.map((s) => ({
            idx: s.idx,
            created_at: s.created_at,
            frame: s.frame,
            angles: s.angles,
            score: s.score ?? null,
          })),
          lang: "ko",
        };

        const report = await generateLLMReport(requestPayload);
        renderLLMReport(report);
        
        // ✅ (B) 버튼을 누를 때마다 "현재 angles" 기준으로 점수를 다시 계산해 반영
        // 종합 리포트 페이지에서는 점수를 LLM이 아니라 프론트 규칙(computeScoreFromAngles)로 산출합니다.
        setScore(computeScoreFromAngles(angles));

        // 요약 문구는 LLM 결과를 그대로 사용(점수는 반영하지 않음)
        const extracted = extractScoreAndSummary(report);
        if (extracted.sub || extracted.headline) setSummaryText(extracted);
      } catch (e) {
        alert(e.message);
      }
    });
  }
})();