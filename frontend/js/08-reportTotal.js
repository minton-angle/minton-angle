const API_BASE = "http://localhost:8000"; // FastAPI 주소

// ====== GT Profile (종합 리포트 페이지에서는 사용 안 함) ======
// NOTE: 종합 페이지는 세션 히스토리 기반이므로, GT 범위는 서버/다른 페이지에서 처리하는 것을 권장합니다.


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

// KF 오차 기반 점수 산출
function computeScoreFromKfError(kfError) {
  const meanAbs = Math.abs(Number(kfError));
  if (!Number.isFinite(meanAbs)) return 0;
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

function renderTableFromSession(session) {
  const tbody = document.querySelector("#anglesTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const rows = [];

  // 우선순위 1) KF1~KF3 오차가 있으면 3줄로 표시
  const kf1 = session?.kf1_error;
  const kf2 = session?.kf2_error;
  const kf3 = session?.kf3_error;

  if ([kf1, kf2, kf3].some((v) => Number.isFinite(Number(v)))) {
    rows.push(["KF1_ERROR", kf1]);
    rows.push(["KF2_ERROR", kf2]);
    rows.push(["KF3_ERROR", kf3]);
  } else {
    // 우선순위 2) 종합 kf_error만 있으면 1줄로 표시
    rows.push(["KF_ERROR(ALL)", session?.kf_error]);
  }

  rows.forEach(([label, val]) => {
    const num = Number(val);
    const sev = severityOf(num);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${label}</td>
      <td>${Number.isFinite(num) ? num.toFixed(2) : "-"}</td>
      <td>${sev}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ====== Chart.js 렌더링 ======
const charts = { scoreKfHistory: null };

let __ALL_SESSIONS__ = [];
let __KF_FILTER__ = "ALL"; // ALL | KF1 | KF2 | KF3

function normalizeKF(x){ return String(x ?? "").toUpperCase(); }

function getFilteredSessions(all){
  const xs = Array.isArray(all) ? all : [];

  // KF1/KF2/KF3 탭: 해당 KF만 필터
  if (__KF_FILTER__ !== "ALL") {
    return xs.filter((s)=> normalizeKF(s?.frame) === __KF_FILTER__);
  }

  // ALL 탭: 날짜(created_at) 기준으로 KF1/KF2/KF3를 묶어서 평균 1포인트로 만들기
  // - SCORE: 평균
  // - KF ERROR: 평균
  // (NOTE) 세션에 score가 없으면 kf_error로 계산
  const map = new Map();
  for (const s of xs){
    const key = String(s?.created_at ?? "");
    if (!key) continue;

    const score = Number.isFinite(Number(s?.score))
      ? Number(s.score)
      : computeScoreFromKfError(s?.kf_error);

    const err = Number.isFinite(Number(s?.kf_error)) ? Number(s.kf_error) : 0;

    if (!map.has(key)) map.set(key, { created_at: key, scoreSum: 0, errSum: 0, n: 0 });
    const g = map.get(key);
    g.scoreSum += score;
    g.errSum += err;
    g.n += 1;
  }

  // created_at 정렬
  const out = Array.from(map.values())
    .sort((a,b)=>{
      const da = new Date(a.created_at).getTime();
      const db = new Date(b.created_at).getTime();
      if (Number.isFinite(da) && Number.isFinite(db)) return da - db;
      return String(a.created_at).localeCompare(String(b.created_at));
    })
    .map((g, i)=>({
      idx: `ALL-${i+1}`,
      created_at: g.created_at,
      frame: "ALL",
      score: +(g.scoreSum / Math.max(1, g.n)).toFixed(0),
      kf_error: +(g.errSum / Math.max(1, g.n)).toFixed(2),
    }));

  return out;
}

function destroyChart(key) {
  if (charts[key]) {
    charts[key].destroy();
    charts[key] = null;
  }
}

function renderScoreKfHistoryChart(sessions) {
  const ctx = document.getElementById("scoreKfHistoryChart");
  if (!ctx) return;
  destroyChart("scoreKfHistory");

  const xs = getFilteredSessions(sessions);

  const labels = xs.map((s) => formatMonthDay(s.created_at ?? null));

  const scoreValues = xs.map((s) => {
    const direct = Number(s.score);
    if (Number.isFinite(direct)) return direct;
    return computeScoreFromKfError(s?.kf_error);
  });

  // 세션별 KF error(평균 |error|) - kf_error 필드가 있으면 우선 사용
  const kfErrValues = xs.map((s) => {
    if (Number.isFinite(Number(s?.kf_error))) return Number(s.kf_error);
    return 0;
  });

  charts.scoreKfHistory = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "SCORE",
          data: scoreValues,
          pointRadius: 2,
          tension: 0.25,
          yAxisID: "y",
          borderColor: "#10b981", // ✅ 초록색
          backgroundColor: "#10b981",
        },
        {
          label: "KF ERROR",
          data: kfErrValues,
          pointRadius: 2,
          tension: 0.25,
          yAxisID: "y1",
          borderColor: "#f59e0b", // ✅ 주황색
          backgroundColor: "#f59e0b",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "bottom" },
        tooltip: {
          callbacks: {
            label: (c) => {
              if (c.dataset.label === "KF ERROR") return ` ${c.parsed.y}°`;
              return ` ${c.parsed.y} 점`;
            },
          },
        },
      },
      scales: {
        y: {
          position: "left",
          min: 0,
          max: 100,
          ticks: { font: { size: 10 } },
          title: { display: true, text: "SCORE" },
        },
        y1: {
          position: "right",
          beginAtZero: true,
          ticks: { font: { size: 10 } },
          grid: { drawOnChartArea: false },
          title: { display: true, text: "KF ERROR (°)" },
        },
        x: { ticks: { font: { size: 10 } } },
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

function safeDate(d){
  const x = new Date(d);
  return Number.isFinite(x.getTime()) ? x : null;
}

// created_at이 유효한 날짜면 "M.D" 형식으로, 아니면 원래 문자열 그대로 반환
function formatMonthDay(x){
  const d = safeDate(x);
  if (!d) return String(x ?? "-");
  const m = d.getMonth() + 1;
  const day = d.getDate();
  return `${m}.${day}`;
}

// ✅ 평균 |오차| 계산
function meanAbsFromAngles(angles){
  const nums = Object.values(angles || {})
    .map((v) => Math.abs(Number(v)))
    .filter(Number.isFinite);
  if (!nums.length) return null;
  return nums.reduce((a,b)=>a+b,0) / nums.length;
}

// ✅ KF별 평균 오차(bar) + worst KF 텍스트 표시
function renderKFBarChart(sessions){
  const ctx = document.getElementById("kfBarChart");
  if (!ctx) return;
  destroyChart("kf");

  const buckets = { KF1: [], KF2: [], KF3: [] };
  (Array.isArray(sessions) ? sessions : []).forEach((s)=>{
    const f = String(s?.frame ?? "").toUpperCase();
    if (!buckets[f]) return;
    const m = meanAbsFromAngles(s?.angles);
    if (m == null) return;
    buckets[f].push(m);
  });

  const labels = ["KF1","KF2","KF3"];
  const values = labels.map((k)=>{
    const arr = buckets[k];
    if (!arr.length) return 0;
    return +(arr.reduce((a,b)=>a+b,0) / arr.length).toFixed(2);
  });

  // worst(가장 큰 평균오차)
  let worstKF = labels[0];
  let worstVal = values[0];
  for (let i=1;i<labels.length;i++){
    if (values[i] > worstVal){ worstVal = values[i]; worstKF = labels[i]; }
  }

  const hint = document.getElementById("worstKFSummary");
  if (hint){
    hint.textContent = `Worst: ${worstKF} (avg |error| ${worstVal}°)`;
    hint.classList.toggle("is-worst-kf2", worstKF === "KF2");
  }

  charts.kf = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "avg |error|", data: values, borderWidth: 1 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c)=>` ${c.parsed.y}°` } },
      },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { beginAtZero: true, ticks: { font: { size: 10 } } },
      },
      animation: { duration: 800 },
    },
  });
}

// ✅ 요일 분포
function renderWeekdayBarChart(sessions){
  const ctx = document.getElementById("weekdayBarChart");
  if (!ctx) return;
  destroyChart("weekday");

  const labels = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
  const counts = new Array(7).fill(0);
  (Array.isArray(sessions) ? sessions : []).forEach((s)=>{
    const d = safeDate(s?.created_at);
    if (!d) return;
    counts[d.getDay()] += 1;
  });

  charts.weekday = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "sessions", data: counts, borderWidth: 1 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } } },
      },
      animation: { duration: 800 },
    },
  });
}

// ✅ 변화 Top3: 첫 세션 vs 마지막 세션 |error| 변화량(감소=개선)
function renderTop3ChangeChart(sessions){
  const ctx = document.getElementById("top3ChangeChart");
  if (!ctx) return;
  destroyChart("top3");

  const xs = Array.isArray(sessions) ? sessions : [];
  const first = xs.length ? xs[0] : null;
  const last = xs.length ? xs[xs.length-1] : null;
  const a0 = first?.angles || {};
  const a1 = last?.angles || {};

  // joint union
  const joints = Array.from(new Set([...Object.keys(a0), ...Object.keys(a1)]));
  const deltas = joints.map((j)=>{
    const v0 = Math.abs(Number(a0[j]));
    const v1 = Math.abs(Number(a1[j]));
    const n0 = Number.isFinite(v0) ? v0 : null;
    const n1 = Number.isFinite(v1) ? v1 : null;
    if (n0 == null || n1 == null) return { joint: j, delta: 0 };
    // 감소(음수) = 개선, 증가(양수) = 악화
    return { joint: j, delta: +(n1 - n0).toFixed(2) };
  });

  // 절대 변화량 기준 Top3
  const top3 = deltas
    .slice()
    .sort((a,b)=>Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0,3);

  const labels = top3.map((x)=>x.joint);
  const values = top3.map((x)=>x.delta);

  charts.top3 = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Δ |error| (last-first)", data: values, borderWidth: 1 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c)=>` ${c.parsed.y}° (Δ)` } },
      },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { ticks: { font: { size: 10 } } },
      },
      animation: { duration: 800 },
    },
  });
}

function renderMonthlySummary(sessions){
  const el = document.getElementById("monthlySummary");
  if (!el) return;

  const xs = Array.isArray(sessions) ? sessions : [];
  if (!xs.length){
    el.textContent = "-";
    return;
  }

  const first = xs[0];
  const last = xs[xs.length-1];
  const s0 = Number.isFinite(Number(first?.score)) ? Number(first.score) : computeScoreFromAngles(first?.angles || {});
  const s1 = Number.isFinite(Number(last?.score)) ? Number(last.score) : computeScoreFromAngles(last?.angles || {});
  const delta = s1 - s0;

  // worst KF
  const kfBuckets = { KF1: [], KF2: [], KF3: [] };
  xs.forEach((s)=>{
    const f = String(s?.frame ?? "").toUpperCase();
    if (!kfBuckets[f]) return;
    const m = meanAbsFromAngles(s?.angles);
    if (m == null) return;
    kfBuckets[f].push(m);
  });
  const kfMeans = Object.entries(kfBuckets).map(([k,arr])=>{
    if (!arr.length) return { k, v: 0 };
    return { k, v: arr.reduce((a,b)=>a+b,0)/arr.length };
  });
  kfMeans.sort((a,b)=>b.v-a.v);
  const worstKF = kfMeans[0]?.k ?? "-";

  el.innerHTML = `
    이번 달 세션 <b>${xs.length}</b>회 기준 요약입니다.<br/>
    점수 변화: <b>${s0} → ${s1}</b> (Δ ${delta >= 0 ? "+" : ""}${delta})<br/>
    가장 불안정한 구간: <b>${worstKF}</b><br/>
    다음 목표: worst 구간에서 오차각을 줄이는 루틴을 1~2개만 고정해 반복하세요.
  `;
}

// (removed) GT_profile-based best KF feature was removed per product decision.
function renderBestKFInRange(){ /* intentionally removed */ }

let __RANGE_FILTER__ = "7d";
window.__LAST_SESSION__ = null;

function wireRangeTabs(onChange){
  const tabs = Array.from(document.querySelectorAll(".rangeTab"));
  if (!tabs.length) return;

  tabs.forEach((btn)=>{
    btn.addEventListener("click", ()=>{
      const v = String(btn.dataset.range || "7d").toLowerCase();
      __RANGE_FILTER__ = (v === "7d" || v === "1m" || v === "3m" || v === "all") ? v : "7d";

      tabs.forEach((b)=>{
        const active = (b === btn);
        b.classList.toggle("is-active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
      });

      if (typeof onChange === "function") onChange(__RANGE_FILTER__);
    });
  });
}

async function refreshByRange(range){
  const payload = await loadFromDB(range);
  const sessions = Array.isArray(payload?.sessions) ? payload.sessions : [];

  __ALL_SESSIONS__ = sessions;
  const last = sessions.length ? sessions[sessions.length - 1] : null;
  window.__LAST_SESSION__ = last;

  // 차트
  renderScoreKfHistoryChart(sessions);

  // 스냅샷(기간 내 마지막)
  window.__CURRENT_FRAME__ = last?.frame ?? "ALL";
  const meta = last?.meta || {};
  renderMeta({ ...meta, created_at: last?.created_at, idx: last?.idx });
  renderTableFromSession(last);

  const initialScore = Number.isFinite(Number(last?.score))
    ? Number(last.score)
    : computeScoreFromKfError(last?.kf_error);
  setScore(initialScore);

  // GT 문구도 같이 갱신
  // (removed) renderBestKFInRange(sessions);
}

// ====== LLM 리포트 생성 호출 (DB 기반) ======
function getPostIdxFromURL() {
  try {
    const u = new URL(window.location.href);
    return u.searchParams.get("post_idx");
  } catch (_) {
    return null;
  }
}

function getPostIdxFallback() {
  try {
    return localStorage.getItem("post_idx");
  } catch (_) {
    return null;
  }
}

async function generateLLMReportByPostIdx(postIdx, lang = "ko") {
  const url = `${API_BASE}/api/report/post/${encodeURIComponent(postIdx)}?lang=${encodeURIComponent(lang)}`;
  const res = await fetch(url, { method: "POST" });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`LLM report failed: ${res.status} ${text}`);
  }

  const data = await res.json(); // { report: { ... } }
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

  // ===== 성장/정체/일관성/좋아진 점 섹션 =====
  const growthEl = document.getElementById("llmGrowth");
  const plateauEl = document.getElementById("llmPlateau");
  const consEl = document.getElementById("llmConsistency");
  const winsEl = document.getElementById("llmWins");

  // growth
  if (growthEl) {
    const g = reportObj?.growth || null;
    const dir = g?.direction ? String(g.direction) : "-";
    const dlt = Number(g?.delta_mean_abs_kf_error);
    const dltText = Number.isFinite(dlt) ? `${dlt >= 0 ? "+" : ""}${dlt}°` : "-";
    const msg = g?.message ? String(g.message) : "-";
    growthEl.innerHTML = `방향: <b>${dir}</b><br/>변화(평균 오차): <b>${dltText}</b><br/>${msg}`;
  }

  // plateau
  if (plateauEl) {
    const p = reportObj?.plateau || null;
    const kf = p?.kf ? String(p.kf) : "-";
    const msg = p?.message ? String(p.message) : "-";
    const why = p?.why ? String(p.why) : "";
    const fixes = Array.isArray(p?.fix) ? p.fix : [];
    const fixText = fixes.length ? fixes.map((x) => `• ${String(x)}`).join("<br/>") : "-";
    plateauEl.innerHTML = `대상: <b>${kf}</b><br/>${msg}${why ? `<br/><br/><b>왜 중요?</b><br/>${why}` : ""}<br/><br/><b>추천 교정</b><br/>${fixText}`;
  }

  // consistency
  if (consEl) {
    const c = reportObj?.consistency || null;
    const kf = c?.kf ? String(c.kf) : "-";
    const msg = c?.message ? String(c.message) : "-";
    const how = Array.isArray(c?.how_to_practice) ? c.how_to_practice : [];
    const howText = how.length ? how.map((x) => `• ${String(x)}`).join("<br/>") : "-";
    consEl.innerHTML = `대상: <b>${kf}</b><br/>${msg}<br/><br/><b>연습 방법</b><br/>${howText}`;
  }

  // wins
  if (winsEl) {
    const wins = Array.isArray(reportObj?.wins) ? reportObj.wins : [];
    if (!wins.length) {
      winsEl.textContent = "-";
    } else {
      winsEl.innerHTML = wins
        .map((w) => {
          const kf = w?.kf ? String(w.kf) : "-";
          const msg = w?.message ? String(w.message) : "-";
          return `• <b>${kf}</b>: ${msg}`;
        })
        .join("<br/>");
    }
  }

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
}

// ====== 초기 로드: DB에서 JSON 가져오기 (실제 DB) ======
async function loadFromDB(range = "7d") {
  const postIdx = getPostIdxFromURL() || getPostIdxFallback();
  if (!postIdx) {
    throw new Error("post_idx가 없습니다. URL에 ?post_idx=... 를 붙이세요.");
  }

  const r = (range || "7d").toLowerCase();
  const url = `${API_BASE}/api/report/analysis/post/${encodeURIComponent(postIdx)}?range=${encodeURIComponent(r)}`;
  console.log("[DB FETCH REQUEST]", url);

  const res = await fetch(url, { method: "GET" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`DB fetch failed: ${res.status} ${text}`);
  }

  const data = await res.json();
  console.log("[DB FETCH RESPONSE]", data);
  return data;
}

// ====== 부트스트랩 ======
(async function init() {
  const payload = await loadFromDB(__RANGE_FILTER__);

  const sessions = Array.isArray(payload?.sessions) ? payload.sessions : [];
  const last = sessions.length ? sessions[sessions.length - 1] : null;

  // KF 탭 필터링을 위해 세션을 전역 상태에 저장
  __ALL_SESSIONS__ = sessions;

  // KF 탭 클릭 시 차트를 현재 필터 기준으로 재렌더
  wireRangeTabs(async (r) => {
    try {
      await refreshByRange(r);
    } catch (e) {
      alert(e.message);
    }
  });

  // 종합 리포트 페이지: 세션 히스토리(여러 번의 평가)를 시각화
  renderScoreKfHistoryChart(__ALL_SESSIONS__);

  // 현재 세션(가장 최근) 스냅샷
  window.__CURRENT_FRAME__ = last?.frame ?? "ALL";
  const meta = last?.meta || {};

  // 메타/스냅샷 렌더
  renderMeta({ ...meta, created_at: last?.created_at, idx: last?.idx });

  // 테이블: KF 오차 기반
  renderTableFromSession(last);

  // 초기 점수: 최근 세션의 score 또는 kf_error 기반
  const initialScore = Number.isFinite(Number(last?.score))
    ? Number(last.score)
    : computeScoreFromKfError(last?.kf_error);

  setScore(initialScore);

  // ===== LLM 버튼 =====
  const btn = document.getElementById("btnGenerate");
  if (btn) {
    btn.addEventListener("click", async () => {
      try {
        const postIdx = getPostIdxFromURL() || getPostIdxFallback();
        if (!postIdx) {
          throw new Error("post_idx가 없습니다. URL에 ?post_idx=... 를 붙이거나 localStorage.post_idx를 설정하세요.");
        }

        const report = await generateLLMReportByPostIdx(postIdx, "ko");
        renderLLMReport(report);
        // renderMonthlySummary(sessions); // (removed: no MONTHLY SUMMARY)
        
        // SCORE는 선택된 기간(range)의 최신 세션 기준으로 유지합니다. (LLM 버튼은 점수 링을 변경하지 않음)

        // 요약 문구는 LLM 결과를 그대로 사용(점수는 반영하지 않음)
        const extracted = extractScoreAndSummary(report);
        if (extracted.sub || extracted.headline) setSummaryText(extracted);
      } catch (e) {
        alert(e.message);
      }
    });
  }
})();