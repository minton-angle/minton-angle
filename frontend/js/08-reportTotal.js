function stageKeyFromActionIndex(n){
  // 5-card order: 1 Ready, 2 Rotation, 3 Backswing, 4 Impact, 5 FollowSwing
  if (String(n) === "1") return "1_Ready_Total";
  if (String(n) === "2") return "2_Rotation_Total";
  if (String(n) === "3") return "3_Backswing_Total";
  if (String(n) === "4") return "4_Impact_Total";
  if (String(n) === "5") return "5_FollowSwing_Total";
  return null;
}

function extractStageSeries(sessions, stageKey){
  const key = String(stageKey || "");
  return (Array.isArray(sessions) ? sessions : []).map((s)=>{
    const v = s?.stage_scores?.[key];
    const num = Number(v);
    return Number.isFinite(num) ? num : null;
  });
}
const API_BASE = "http://localhost:8000"; // FastAPI 주소

// ====== GT Profile (종합 리포트 페이지에서는 사용 안 함) ======
// NOTE: 종합 페이지는 세션 히스토리 기반이므로, GT 범위는 서버/다른 페이지에서 처리하는 것을 권장합니다.


// ====== KF 키 → 사용자 친화 명칭 ======
function actionNameFromKfKey(kfKey) {
  const k = String(kfKey || "").toLowerCase();
  if (k.includes("kf1")) return "백스윙";
  if (k.includes("kf2")) return "임팩트";
  if (k.includes("kf3")) return "팔로스윙";
  return "-";
}

function kfFieldFromActionIndex(n){
  // 5-card order: 1 Ready, 2 Rotation, 3 Backswing, 4 Impact, 5 FollowSwing
  // KF error series exists only for cards 3~5.
  if (String(n) === "3") return "kf1_error";
  if (String(n) === "4") return "kf2_error";
  if (String(n) === "5") return "kf3_error";
  return null;
}

function kfKeyOfAction(actionNum){
  return kfFieldFromActionIndex(actionNum);
}

function trendPillText(direction, delta){
  const dir = String(direction || "flat");
  const d = Number(delta);
  const abs = Number.isFinite(d) ? Math.abs(d).toFixed(2) : "-";

  // If delta looks like degrees (legacy path), keep old arrow. Otherwise treat as score.
  // Heuristic: degrees deltas are usually small (< 20). Scores are also < 20, but we prefer score wording here.
  if (dir === "improved") return `개선 (Δ ${abs}점)`;
  if (dir === "worsened") return `악화 (Δ ${abs}점)`;
  return `정체 (Δ ${abs}점)`;
}

function applyTrendPill(el, direction){
  if (!el) return;
  el.classList.remove("is-good","is-bad","is-flat");
  const dir = String(direction || "flat");
  if (dir === "improved") el.classList.add("is-good");
  else if (dir === "worsened") el.classList.add("is-bad");
  else el.classList.add("is-flat");
}

function meanAbs(values){
  const xs = (Array.isArray(values) ? values : [])
    .map((v)=>Math.abs(Number(v)))
    .filter(Number.isFinite);
  if (!xs.length) return null;
  return xs.reduce((a,b)=>a+b,0) / xs.length;
}

function std(values){
  const xs = (Array.isArray(values) ? values : [])
    .map((v)=>Math.abs(Number(v)))
    .filter(Number.isFinite);
  if (xs.length < 2) return 0;
  const m = xs.reduce((a,b)=>a+b,0) / xs.length;
  return Math.sqrt(xs.reduce((a,x)=>a + (x-m)*(x-m), 0) / xs.length);
}

// 팔로우스윙(카드5) 패스 여부 시리즈 추출: prefers followswing_pass, fallback to stage_scores["5_FollowSwing_Total"]
function extractFollowSwingPassSeries(sessions){
  return (Array.isArray(sessions) ? sessions : []).map((s)=>{
    // primary: boolean from backend
    const v = s?.followswing_pass;
    if (v === true) return true;
    if (v === false) return false;

    // fallback: infer from stage score total (0/100)
    const t = Number(s?.stage_scores?.["5_FollowSwing_Total"]);
    if (Number.isFinite(t)) return t >= 50; // 100=>true, 0=>false
    return null;
  });
}

function extractKfSeries(sessions, field){
  return (Array.isArray(sessions) ? sessions : []).map((s)=>s?.[field]);
}

function buildActionInsightText(actionLabel, curMean, prevMean, curStd){
  const cm = Number.isFinite(curMean) ? curMean.toFixed(2) : "-";
  const pm = Number.isFinite(prevMean) ? prevMean.toFixed(2) : "-";
  const sd = Number.isFinite(curStd) ? curStd.toFixed(2) : "-";

  return `평균 점수: <b>${cm}점</b> (이전 ${pm}점)<br/>변동성(표준편차): <b>${sd}점</b>`;
}

function followswingFeedbackFromFalseRate(falseRate){
  // falseRate(0~1)
  // - 40% 미만: 잘하고 있다
  // - 40% 이상 ~ 80% 미만: 개선 필요
  // - 80% 이상: 위험 부상이 있다
  if (!Number.isFinite(falseRate)) return "-";
  if (falseRate >= 0.80) return "위험 부상이 있어요!";
  if (falseRate >= 0.40) return "팔로우 스윙에 개선이 필요해요!";
  return "자세가 좋으시네요! 그대로 유지해주세요!";
}

function followswingTrendPillText(direction, deltaPp){
  const dp = Number(deltaPp);
  const abs = Number.isFinite(dp) ? Math.abs(dp).toFixed(0) : "-";
  if (direction === "improved") return `개선 (Δ ${abs}%p 감소)`;
  if (direction === "worsened") return `악화 (Δ ${abs}%p 증가)`;
  return `정체 (Δ ${abs}%p)`;
}

function renderActionCards(currentSessions, prevSessions){
  const cards = [
    { n: 1, label: "준비" },
    { n: 2, label: "회전" },
    { n: 3, label: "백스윙" },
    { n: 4, label: "임팩트" },
    { n: 5, label: "팔로스윙" },
  ];

  let worst = { n: 3, mean: -1 };
  let volatile = { n: 3, std: -1 };

  for (const c of cards){
    const stageKey = stageKeyFromActionIndex(c.n);

    const curArr = extractStageSeries(currentSessions, stageKey);
    const prevArr = extractStageSeries(prevSessions, stageKey);

    const curMean = meanAbs(curArr);
    const prevMean = meanAbs(prevArr);
    const curStd = std(curArr);

    let direction = "flat";
    let delta = null;
    if (Number.isFinite(curMean) && Number.isFinite(prevMean)){
      delta = curMean - prevMean;
      if (delta > 1e-9) direction = "improved";
      else if (delta < -1e-9) direction = "worsened";
    }

    // 카드 5: 성공률 기반 게이지, 그 외는 기존대로 stage mean 사용
    let actionScore;
    if (String(c.n) === "5"){
      // success rate = 100 - falseRate%
      const curSeries = extractFollowSwingPassSeries(currentSessions);
      const curValid = curSeries.filter((v)=> v === true || v === false);
      const curTotalN = curValid.length;
      const curFalseN = curValid.filter((v)=> v === false).length;
      const curFalseRate = curTotalN ? (curFalseN / curTotalN) : 0;

      const successRate = 100 - Math.round(curFalseRate * 100);
      actionScore = clamp(successRate, 0, 100);

      // 성공률은 높을수록 좋으므로 direction은 기존 점수 direction 유지
      setHalfGauge(c.n, actionScore, direction);
    } else {
      actionScore = Number.isFinite(curMean) ? Math.round(clamp(curMean, 0, 100)) : 0;
      setHalfGauge(c.n, actionScore, direction);
    }

    // Per-card mini stage history chart (cards 1~4): stage_scores overlay (current vs previous)
    if (String(c.n) !== "5"){
      renderActionMiniStageChart(c.n, curArr, prevArr, __RANGE_FILTER__);
    }

    // worst: lowest mean score
    if (Number.isFinite(curMean) && (worst.mean < 0 || curMean < worst.mean)){
      worst = { n: c.n, mean: curMean };
    }
    // volatile: largest std
    if (Number.isFinite(curStd) && curStd > volatile.std){
      volatile = { n: c.n, std: curStd };
    }

    // special case: card 5 (FollowSwing) feedback
    const body = document.getElementById(`a${c.n}Body`);
    if (!body) continue;

    // FollowSwing: boolean-based feedback (false rate) + previous period comparison
    if (String(c.n) === "5"){
      const curSeries = extractFollowSwingPassSeries(currentSessions);
      const curValid = curSeries.filter((v)=> v === true || v === false);
      const curTotalN = curValid.length;
      const curFalseN = curValid.filter((v)=> v === false).length;
      const curFalseRate = curTotalN ? (curFalseN / curTotalN) : NaN;

      const prevSeries = extractFollowSwingPassSeries(prevSessions);
      const prevValid = prevSeries.filter((v)=> v === true || v === false);
      const prevTotalN = prevValid.length;
      const prevFalseN = prevValid.filter((v)=> v === false).length;
      const prevFalseRate = prevTotalN ? (prevFalseN / prevTotalN) : NaN;

      const curPct = Number.isFinite(curFalseRate) ? Math.round(curFalseRate * 100) : null;
      const prevPct = Number.isFinite(prevFalseRate) ? Math.round(prevFalseRate * 100) : null;

      const deltaPp = (Number.isFinite(curFalseRate) && Number.isFinite(prevFalseRate))
        ? Math.round((curFalseRate - prevFalseRate) * 100)
        : null;

      // false rate는 낮을수록 좋음
      let fsDir = "flat";
      if (deltaPp != null){
        if (deltaPp < 0) fsDir = "improved";
        else if (deltaPp > 0) fsDir = "worsened";
      }

      // Half gauge: 성공률(%)
      const successRate = 100 - Math.round((Number.isFinite(curFalseRate) ? curFalseRate : 0) * 100);
      const actionScore = clamp(successRate, 0, 100);
      setHalfGauge(c.n, actionScore, fsDir);

      const msg = followswingFeedbackFromFalseRate(curFalseRate);
      body.innerHTML = `기간 내 팔로우 스윙을 못한 비율: <b>${curPct == null ? "-" : curPct + "%"}</b> (False ${curFalseN}/${curTotalN})<br/>` +
                       `이전 기간내 못한 비율: <b>${prevPct == null ? "-" : prevPct + "%"}</b> (False ${prevFalseN}/${prevTotalN})<br/>` +
                       `변화: <b>${deltaPp == null ? "-" : (deltaPp > 0 ? "+" : "") + deltaPp + "%p"}</b><br/>` +
                       `<b>${msg}</b>`;

      // pill: reflect false-rate delta
      const pillEl = document.getElementById(`a${c.n}TrendPill`);
      if (pillEl){
        pillEl.textContent = followswingTrendPillText(fsDir, deltaPp == null ? NaN : deltaPp);
        applyTrendPill(pillEl, fsDir);
      }
    } else {
      body.innerHTML = buildActionInsightText(c.label, curMean, prevMean, curStd);
      const pill = document.getElementById(`a${c.n}TrendPill`);
      if (pill){
        pill.textContent = trendPillText(direction, delta);
        applyTrendPill(pill, direction);
      }
    }
  }

  // 추천 영상: 평균 점수 worst + 편차 worst 기반 자동 큐레이션 (stage keys)
  const worstKey = stageKeyFromActionIndex(worst.n);
  const volKey = stageKeyFromActionIndex(volatile.n);
  const keys = [worstKey, volKey].filter(Boolean);
  renderYoutubeLinksByKfKeys(keys);

  // FollowSwing 성공/실패율 도넛(현재 기간)
  renderFollowSwingDonutCurrent(currentSessions);
}

// ====== Half gauge helpers (per action card) ======
const HALF_GAUGE_DASH = 157; // approx half circumference for the SVG arc

function setHalfGauge(n, score, direction){
  const v = clamp(Math.round(Number(score) || 0), 0, 100);
  const fg = document.getElementById(`a${n}GaugeFg`);
  const num = document.getElementById(`a${n}GaugeValue`);
  if (num) num.textContent = String(v);

  if (fg){
    // 0 -> full offset (empty), 100 -> 0 (full)
    const off = HALF_GAUGE_DASH * (1 - v / 100);
    fg.style.strokeDashoffset = String(off);

    const dir = String(direction || "flat").toLowerCase();
    if (dir === "improved") fg.style.stroke = "#22c55e";
    else if (dir === "worsened") fg.style.stroke = "#ef4444";
    else fg.style.stroke = "#6b7280";
  }
}

function computeActionScoreFromMean(meanErr){
  // meanErr is degrees (already absolute mean). Reuse existing KF score mapping.
  return computeScoreFromKfError(meanErr);
}

// ====== 동작별 피드백 카드 렌더 ======

function setText(id, html){
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = (html == null || html === "") ? "" : String(html);
}

// ===== A안: 동작별 간단 구조 =====
// 새 actions 필드 기반 간단 카드 렌더링
function renderActionSimple(n, key) {
  const actions = window.__LLM_ACTIONS__ || {};
  const a = actions[key] || {};
  const title = a?.title || "-";
  const problem = a?.problem_one || "-";
  const fixes = Array.isArray(a?.fix_two) ? a.fix_two : [];
  const fixText = fixes.length ? fixes.map((x)=>`• ${String(x)}`).join("<br/>") : "-";

  setText(`a${n}Summary`, `<b>${title}</b>`);
  setText(`a${n}Plateau`, problem);
  setText(`a${n}Consistency`, fixText);
  setText(`a${n}Wins`, "-");
  setText(`a${n}Issues`, "-");
  setText(`a${n}Checklist`, "-");
}



// ====== 동작 카드 가로 캐러셀(스냅) 보조 ======
function wireActionCarousel(){
  const carousel = document.querySelector(".actionCards");
  if (!carousel) return;

  // 카드 폭(스크롤 스냅 기준) 추정: 첫 카드의 bounding box 사용
  function panelWidth(){
    const first = carousel.querySelector(".actionCard");
    if (!first) return 0;
    const rect = first.getBoundingClientRect();
    // gap 포함을 위해 다음 카드의 left 차이를 우선 사용
    const second = first.nextElementSibling;
    if (second && second.classList.contains("actionCard")){
      const r2 = second.getBoundingClientRect();
      const w = Math.abs(r2.left - rect.left);
      if (Number.isFinite(w) && w > 0) return w;
    }
    return rect.width;
  }

  function getCurrentAction(){
    const w = panelWidth();
    if (!w) return 1;
    const idx = Math.round(carousel.scrollLeft / w);
    return idx + 1;
  }

  // (선택) 현재 카드 인덱스를 body에 data로만 남겨둠 — UI 토글 로직이 따로 없어도 에러 없이 동작
  function setActive(n){
    carousel.dataset.active = String(n);
  }

  // update active on swipe/scroll end
  let t = null;
  carousel.addEventListener("scroll", ()=>{
    if (t) clearTimeout(t);
    t = setTimeout(()=>{
      const n = getCurrentAction();
      setActive(n);
    }, 120);
  }, { passive: true });

  // 초기 active 설정
  setActive(1);
}


// ====== Curated YouTube 추천 (검색이 아니라, 특정 영상 ID 기반) ======
function youtubeWatchUrl(videoId) {
  const id = String(videoId || "").trim();
  if (!id) return null;
  return `https://www.youtube.com/watch?v=${encodeURIComponent(id)}`;
}

function youtubeThumbUrl(videoId) {
  const id = String(videoId || "").trim();
  if (!id) return null;
  return `https://i.ytimg.com/vi/${encodeURIComponent(id)}/hqdefault.jpg`;
}

// 제품에서 큐레이션한 목록만 사용 (채널/영상 ID 고정)
const CURATED_YT = {
  // legacy KF buckets
  action1: [
    { videoId: "toQ7tOx7Tvs", title: "The 4 Grips In Badminton (올바른 그립 전환)", channel: "Badminton Insight" },
    { videoId: "xRv1JLg4NMM", title: "Forehand Overhead Clear Tutorial (준비/타이밍)", channel: "Badminton Insight" },
  ],
  action2: [
    { videoId: "H7kpZ9inc10", title: "Badminton SMASH Tutorial (파워와 타이밍)", channel: "Badminton Insight" },
  ],
  action3: [
    { videoId: "zCq36gnqGdI", title: "How To Use Your Wrist In Badminton (손목이 아니라 손가락)", channel: "Badminton Insight" },
  ],

  // stage buckets (A안)
  ready: [
    { videoId: "xRv1JLg4NMM", title: "Forehand Overhead Clear Tutorial (준비/타이밍)", channel: "Badminton Insight" },
    { videoId: "toQ7tOx7Tvs", title: "The 4 Grips In Badminton (그립/준비)", channel: "Badminton Insight" },
  ],
  rotation: [
    { videoId: "H7kpZ9inc10", title: "Badminton SMASH Tutorial (회전/타이밍)", channel: "Badminton Insight" },
  ],
  backswing: [
    { videoId: "xRv1JLg4NMM", title: "Forehand Overhead Clear Tutorial (백스윙 연결)", channel: "Badminton Insight" },
  ],
  impact: [
    { videoId: "H7kpZ9inc10", title: "Badminton SMASH Tutorial (임팩트 포인트)", channel: "Badminton Insight" },
  ],
  followswing: [
    { videoId: "zCq36gnqGdI", title: "How To Use Your Wrist In Badminton (팔로스윙/손가락)", channel: "Badminton Insight" },
  ],
};

function curatedListForKf(kfKey) {
  const raw = String(kfKey || "");
  const k = raw.toLowerCase();

  // stage keys
  if (k.includes("1_ready_total") || k === "ready") return CURATED_YT.ready;
  if (k.includes("2_rotation_total") || k === "rotation") return CURATED_YT.rotation;
  if (k.includes("3_backswing_total") || k === "backswing") return CURATED_YT.backswing;
  if (k.includes("4_impact_total") || k === "impact") return CURATED_YT.impact;
  if (k.includes("5_followswing_total") || k === "followswing") return CURATED_YT.followswing;

  // legacy KF keys
  if (k.includes("kf1")) return CURATED_YT.action1;
  if (k.includes("kf2")) return CURATED_YT.action2;
  if (k.includes("kf3")) return CURATED_YT.action3;

  return [];
}

function uniqByVideoId(items) {
  const seen = new Set();
  const out = [];
  for (const it of Array.isArray(items) ? items : []) {
    const id = String(it?.videoId || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push(it);
  }
  return out;
}

function renderYoutubeLinksByKfKeys(kfKeys) {
  const wrap = document.getElementById("llmYoutubeLinks");
  if (!wrap) return;

  const keys = Array.isArray(kfKeys) ? kfKeys : [];
  const picked = [];
  keys.forEach((k)=> picked.push(...curatedListForKf(k)));

  if (!picked.length) {
    picked.push(...CURATED_YT.action1, ...CURATED_YT.action2, ...CURATED_YT.action3);
  }

  const list = uniqByVideoId(picked).slice(0, 3);

  if (!list.length) {
    wrap.innerHTML = `<div class="ytEmpty"></div>`;
    return;
  }

  wrap.innerHTML = list
    .map((v) => {
      const url = youtubeWatchUrl(v.videoId);
      const thumb = youtubeThumbUrl(v.videoId);
      const title = String(v.title || "추천 영상");
      const channel = String(v.channel || "");

      return `
        <a class="ytCard" href="${url}" target="_blank" rel="noopener noreferrer">
          <div class="ytCard__thumbWrap">
            <img class="ytCard__thumb" src="${thumb}" alt="${title}" loading="lazy" />
            <div class="ytCard__badge">YouTube</div>
          </div>
          <div class="ytCard__body">
            <div class="ytCard__title">${title}</div>
            <div class="ytCard__meta">${channel}</div>
          </div>
        </a>
      `;
    })
    .join("");
}

function renderYoutubeTableFromReport(reportObj) {
  const tbody = document.querySelector("#ytTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const plateauKf = reportObj?.plateau?.kf || null;
  const consKf = reportObj?.consistency?.kf || null;

  const picked = [];
  if (plateauKf) picked.push({ kf: plateauKf, list: curatedListForKf(plateauKf) });
  if (consKf && consKf !== plateauKf) picked.push({ kf: consKf, list: curatedListForKf(consKf) });

  // fallback: 대표 세트
  if (!picked.length) {
    picked.push({ kf: "kf1_error", list: CURATED_YT.action1 });
    picked.push({ kf: "kf2_error", list: CURATED_YT.action2 });
    picked.push({ kf: "kf3_error", list: CURATED_YT.action3 });
  }

  const rows = [];
  for (const group of picked) {
    for (const v of Array.isArray(group.list) ? group.list : []) {
      rows.push({
        action: actionNameFromKfKey(group.kf),
        title: String(v?.title || "추천 영상"),
        channel: String(v?.channel || ""),
        url: youtubeWatchUrl(v?.videoId),
      });
    }
  }

  const uniq = [];
  const seen = new Set();
  for (const r of rows) {
    if (!r.url || seen.has(r.url)) continue;
    seen.add(r.url);
    uniq.push(r);
  }

  const top = uniq.slice(0, 6);
  if (!top.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4">-</td>`;
    tbody.appendChild(tr);
    return;
  }

  for (const r of top) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.action}</td>
      <td title="${r.title}">${r.title}</td>
      <td title="${r.channel}">${r.channel || "-"}</td>
      <td><a href="${r.url}" target="_blank" rel="noopener noreferrer">열기</a></td>
    `;
    tbody.appendChild(tr);
  }
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

// ====== 기간 라벨/성장 요약/스코어 링 색상 헬퍼 ======
function rangeLabelFromKey(r){
  const x = String(r || "").toLowerCase();
  if (x === "7d") return "1주일";
  if (x === "1m") return "1개월";
  if (x === "3m") return "3개월";
  if (x === "all") return "전체";
  return "기간";
}

// 성장 방향에 따라 Score Ring 색상 변경
function setScoreRingColor(direction){
  const dir = String(direction || "").toLowerCase();
  const fg = document.getElementById("scoreRingFg");
  if (!fg) return;

  // 기존 컬러 톤 유지: 개선=초록, 악화=빨강, 정체=회색
  if (dir === "improved") fg.style.stroke = "#22c55e";
  else if (dir === "worsened") fg.style.stroke = "#ef4444";
  else fg.style.stroke = "#6b7280";
}

// comparison 기반 “지난 기간 대비” 문구 렌더
function renderGrowthSummary(comparison, rangeKey){
  const el = document.getElementById("summarySub");
  if (!el) return;

  const label = rangeLabelFromKey(rangeKey);
  if (!comparison) {
    el.textContent = `${label} 기준 분석 결과입니다.`;
    return;
  }

  const dir = String(comparison.direction || "flat");
  const dlt = Number(comparison.delta_mean_abs_kf_error);
  const abs = Number.isFinite(dlt) ? Math.abs(dlt).toFixed(2) : null;

  if (dir === "improved" && abs != null) {
    el.innerHTML = `지난 ${label} 대비 평균 오차가 </br> <b style="color:#16a34a">${abs}° 감소</b>했습니다.<br/>꾸준한 훈련이 유지되고 있어요!`;
  } else if (dir === "worsened" && abs != null) {
    el.innerHTML = `지난 ${label} 대비 평균 오차가 </br> <b style="color:#b91c1c">${abs}° 증가</b>했습니다.<br/> 훈련에 좀더 집중해 보아요!`;
  } else {
    el.textContent = `지난 ${label} 대비 변화가 거의 없군요. 유지하는 것도 좋은 현상입니다!`;
  }
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
    rows.push(["동작 1", kf1]);
    rows.push(["동작 2", kf2]);
    rows.push(["동작 3", kf3]);
    rows.push(["동작 1", kf1]);
    rows.push(["동작 2", kf2]);
    rows.push(["동작 3", kf3]);
  } else {
    // 우선순위 2) 종합 kf_error만 있으면 1줄로 표시
    rows.push(["평균(전체)", session?.kf_error]);
    rows.push(["평균(전체)", session?.kf_error]);
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
const charts = {
  scoreKfHistory: null,
  // per-action mini charts: actionMini1..4
  actionMini1: null,
  actionMini2: null,
  actionMini3: null,
  actionMini4: null,
  // FollowSwing은 boolean 기반이므로 mini chart 제외, 대신 도넛 차트로 성공/실패 비율 표시
  followSwingDonut: null
};

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
  if (charts && charts[key]) {
    try { charts[key].destroy(); } catch (_) {}
    charts[key] = null;
  }
}

function renderActionMiniStageChart(actionNum, currentSeries, prevSeries, rangeKey){
  const n = String(actionNum);
  if (n === "5") return; // FollowSwing은 boolean 기반: 미니차트 제외

  const canvas = document.getElementById(`a${n}StageChart`);
  if (!canvas) return;

  const chartKey = `actionMini${n}`;
  destroyChart(chartKey);

  const r = String(rangeKey || "7d").toLowerCase();
  let N = Array.isArray(currentSeries) ? currentSeries.length : 0;
  if (r === "7d") N = 7;
  else if (r === "1m") N = 30;
  else if (r === "3m") N = 90;
  else if (r === "all") N = Array.isArray(currentSeries) ? currentSeries.length : 0;

  N = clamp(Number(N) || 0, 1, 120);
  const labels = Array.from({ length: N }, (_, i) => String(i + 1));

  function alignLastN(arr){
    const xs = Array.isArray(arr) ? arr : [];
    const tail = xs.slice(-N);
    const pad = Array(Math.max(0, N - tail.length)).fill(null);
    return pad.concat(tail).map((v)=>{
      const num = Number(v);
      return Number.isFinite(num) ? num : null;
    });
  }

  const cur = alignLastN(currentSeries);
  const prev = alignLastN(prevSeries);

  // y축 자동 확대
  const allVals = [...cur, ...prev]
    .filter((v)=> Number.isFinite(Number(v)))
    .map(Number);

  let yMin = 0;
  let yMax = 100;

  if (allVals.length){
    const vMin = Math.min(...allVals);
    const vMax = Math.max(...allVals);

    const pad = 5;
    yMin = Math.floor((vMin - pad) / 5) * 5;
    yMax = Math.ceil((vMax + pad) / 5) * 5;

    yMin = clamp(yMin, 0, 100);
    yMax = clamp(yMax, 0, 100);

    if (yMax - yMin < 10) yMax = clamp(yMin + 10, 0, 100);
    if (yMax <= yMin){ yMin = 0; yMax = 100; }
  }

  charts[chartKey] = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "현재",
          data: cur,
          backgroundColor: "rgba(34,197,94,0.7)",
          borderRadius: 4,
          barPercentage: 0.9,
          categoryPercentage: 0.5,
        },
        {
          label: "이전",
          data: prev,
          backgroundColor: "rgba(249,115,22,0.7)",
          borderRadius: 4,
          barPercentage: 0.9,
          categoryPercentage: 0.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        x: {
          stacked: false,
          grid: { display: false },
          ticks: { display: false },
        },
        y: {
          min: yMin,
          max: yMax,
          grid: { display: false },
          ticks: { display: false },
        },
      },
    },
  });
}

function renderScoreKfHistoryChart(currentSessions, prevSessions) {
  const ctx = document.getElementById("scoreKfHistoryChart");
  if (!ctx) return;
  destroyChart("scoreKfHistory");

  const cur = getFilteredSessions(currentSessions);
  const prev = getFilteredSessions(prevSessions);

  // ---- Index-based overlay (1..N points) ----
  const rangeKey = (__RANGE_FILTER__ || "7d");
  const prevLabel = rangeLabelFromKey(rangeKey);

  // Choose N by selected range (fallback to current length)
  let N = cur.length;
  if (rangeKey === "7d") N = 7;
  else if (rangeKey === "1m") N = 30;
  else if (rangeKey === "3m") N = 90;
  else if (rangeKey === "all") N = cur.length;

  // Safety cap to avoid huge charts
  N = clamp(Number(N) || 0, 1, 120);

  // Build labels as simple indices
  const labels = Array.from({ length: N }, (_, i) => String(i + 1));

  // Take last N points and align to indices (pad front with null if shorter)
  function alignLastN(xs, pick) {
    const arr = (Array.isArray(xs) ? xs : []).map(pick);
    const tail = arr.slice(-N);
    const pad = Array(Math.max(0, N - tail.length)).fill(null);
    return pad.concat(tail);
  }

  const curScore = alignLastN(cur, (s) => {
    const direct = Number(s?.score);
    if (Number.isFinite(direct)) return direct;
    return computeScoreFromKfError(s?.kf_error);
  });

  const prevScore = alignLastN(prev, (s) => {
    const direct = Number(s?.score);
    if (Number.isFinite(direct)) return direct;
    return computeScoreFromKfError(s?.kf_error);
  });

  // Dynamic Y zoom (based on visible values from both series)
  const allVals = [...curScore, ...prevScore].filter((v) => Number.isFinite(Number(v))).map(Number);
  let yMin = 0;
  let yMax = 100;
  if (allVals.length) {
    const vMin = Math.min(...allVals);
    const vMax = Math.max(...allVals);
    // add padding and snap to 5-point steps
    const pad = 5;
    yMin = Math.floor((vMin - pad) / 5) * 5;
    yMax = Math.ceil((vMax + pad) / 5) * 5;

    // ensure a minimum visible range
    if (yMax - yMin < 10) {
      const mid = (yMax + yMin) / 2;
      yMin = Math.floor((mid - 5) / 5) * 5;
      yMax = Math.ceil((mid + 5) / 5) * 5;
    }

    // clamp to score domain
    yMin = clamp(yMin, 0, 100);
    yMax = clamp(yMax, 0, 100);

    // if equal after clamp, fallback
    if (yMax <= yMin) {
      yMin = 0;
      yMax = 100;
    }
  }

  charts.scoreKfHistory = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        // current
        // current
        {
          label: "SCORE",
          data: curScore,
          data: curScore,
          pointRadius: 2,
          tension: 0.25,
          yAxisID: "y",
          borderColor: "#10b981",
          borderColor: "#10b981",
          backgroundColor: "#10b981",
          spanGaps: true,
          spanGaps: true,
        },
        // previous (orange overlay)
        {
          label: `SCORE(${prevLabel} 전)`,
          data: prevScore,
          pointRadius: 2,
          tension: 0.25,
          yAxisID: "y",
          borderColor: "rgba(249,115,22,0.6)",
          backgroundColor: "rgba(249,115,22,0.6)",
          spanGaps: true,
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
            label: (c) => ` ${c.parsed.y} 점`
          },
        },
      },
      scales: {
        y: {
          position: "left",
          min: yMin,
          max: yMax,
          ticks: { font: { size: 10 } },
          title: { display: false, text: "" },
          grid: { display: false },
        },
        x: {
          grid: { display: false },
          ticks: {
            font: { size: 10 },
            callback: (val, idx) => labels[idx]
          }
        },
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

// ✅ FollowSwing 성공/실패율(%) 도넛 (현재 기간만)
function computeFollowSwingRates(sessions){
  const series = extractFollowSwingPassSeries(sessions);
  const valid = series.filter((v)=> v === true || v === false);
  const total = valid.length;
  const fail = valid.filter((v)=> v === false).length;
  const success = valid.filter((v)=> v === true).length;

  if (!total) {
    return { total: 0, success: 0, fail: 0, successPct: 0, failPct: 0 };
  }

  const successPct = Math.round((success / total) * 100);
  const failPct = 100 - successPct;
  return { total, success, fail, successPct, failPct };
}

function renderFollowSwingDonutCurrent(currentSessions){
  const ctx = document.getElementById("followSwingDonutChart");
  if (!ctx) return;

  destroyChart("followSwingDonut");

  const r = computeFollowSwingRates(currentSessions);

  // Center text plugin (Chart.js v3+)
  const centerTextPlugin = {
    id: "centerTextPlugin",
    afterDraw(chart){
      const { ctx } = chart;
      const meta = chart.getDatasetMeta(0);
      const center = meta?.data?.[0];
      if (!center) return;

      const x = center.x;
      const y = center.y;

      ctx.save();
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      // main
      ctx.font = "700 18px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
      ctx.fillStyle = "rgba(17,24,39,.92)";
      ctx.fillText(`${r.successPct}%`, x, y - 2);

      // sub
      ctx.font = "600 11px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
      ctx.fillStyle = "rgba(17,24,39,.55)";
      ctx.fillText(`성공 (${r.success}/${r.total})`, x, y + 16);

      ctx.restore();
    }
  };

  charts.followSwingDonut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["성공", "실패"],
      datasets: [
        {
          label: "FollowSwing",
          data: [r.successPct, r.failPct],
          backgroundColor: ["rgba(34,197,94,0.85)", "rgba(239,68,68,0.75)"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "70%",
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: { boxWidth: 10, font: { size: 10 } },
        },
        tooltip: {
          callbacks: {
            label: (c)=> ` ${c.label}: ${c.parsed}%`,
          }
        }
      },
      animation: { duration: 700 },
    },
    plugins: [centerTextPlugin],
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

// 
function computeScoreFromAngles(angles){
  const m = meanAbsFromAngles(angles);
  if (m == null) return 0;
  // 각도 평균오차를 kfError로 간주하여 기존 점수 매핑 사용
  return computeScoreFromKfError(m);
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
  // 기간 탭(1주/1개월/3개월/전체)만 바인딩: LLM 생성 버튼(btnGenerateLLM)이 실수로 같이 묶이지 않도록 범위를 제한
  const tabs = Array.from(document.querySelectorAll(".rangeTabs .rangeTab[data-range]"));
  if (!tabs.length) return;

  tabs.forEach((btn)=>{
    btn.addEventListener("click", ()=>{
      // 기간 탭만 바인딩 (btnGenerateLLM 등은 제외)
      const rawRange = btn.getAttribute("data-range");
      if (!rawRange) return; // safety
      const v = String(rawRange || "7d").toLowerCase();
      __RANGE_FILTER__ = (v === "7d" || v === "1m" || v === "3m" || v === "all") ? v : "7d";

      tabs.forEach((b)=>{
        const active = (b === btn);
        b.classList.toggle("is-active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
      });

      if (typeof onChange === "function") onChange(__RANGE_FILTER__);
      if (typeof onChange === "function") onChange(__RANGE_FILTER__);
    });
  });
}

async function refreshByRange(range){
  const payload = await loadFromDB(range);

  const current = Array.isArray(payload?.current_sessions) ? payload.current_sessions : [];
  const prev = Array.isArray(payload?.prev_sessions) ? payload.prev_sessions : [];
  const comp = payload?.comparison || null;

  __ALL_SESSIONS__ = current;
  const last = current.length ? current[current.length - 1] : null;
  window.__LAST_SESSION__ = last;

  // 상단 성장 문구 + 링 색상
  renderGrowthSummary(comp, payload?.range || range);
  setScoreRingColor(comp?.direction);

  // 차트: current + prev(점선)
  renderScoreKfHistoryChart(current, prev);

  // KF별 분석 차트
  renderActionCards(current, prev);

  // 스냅샷(기간 내 마지막)
  window.__CURRENT_FRAME__ = last?.frame ?? "ALL";
  const meta = last?.meta || {};
  renderMeta({ ...meta, created_at: last?.created_at, idx: last?.idx });

  const initialScore = Number.isFinite(Number(last?.score))
    ? Number(last.score)
    : computeScoreFromKfError(last?.kf_error);
  setScore(initialScore);

  // If server provides the latest LLM report, render it (doesn't affect charts)
  const latestLLM = payload?.latest_llm_report?.report || null;
  if (latestLLM) {
    renderLLMReport(latestLLM);
  }
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

// ====== LLM 리포트 생성 (기존 라우터: /api/report/post/{post_idx}) ======
// ====== LLM 리포트 생성 (기존 라우터: /api/report/post/{post_idx}) ======
async function generateLLMReportByPostIdx(postIdx, lang = "ko") {
  const r = (__RANGE_FILTER__ || "7d");
  const url = `${API_BASE}/api/report/post/${encodeURIComponent(postIdx)}?lang=${encodeURIComponent(lang)}&range=${encodeURIComponent(r)}`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`LLM report failed: ${res.status} ${text}`);
  }

  const data = await res.json(); // { report: { ... } } 또는 { ... }
  return data?.report ?? data;
}

// LLM report -> 추천 영상 선택(plateau/consistency 기반)
function renderYoutubeLinksFromReport(reportObj){
  const plateauKf = reportObj?.plateau?.kf || reportObj?.plateau?.key || null;
  const consKf = reportObj?.consistency?.kf || reportObj?.consistency?.key || null;

  // score-based report fallback: use sections to choose videos
  // backswing -> kf1, impact -> kf2, followswing -> kf3
  const hasSections = reportObj && typeof reportObj === "object" && reportObj.sections && typeof reportObj.sections === "object";
  const sectionFallbackKeys = hasSections ? [
    "1_Ready_Total",
    "2_Rotation_Total",
    "3_Backswing_Total",
    "4_Impact_Total",
    "5_FollowSwing_Total",
  ] : [];

  const keys = [];
  if (plateauKf) keys.push(plateauKf);
  if (consKf && consKf !== plateauKf) keys.push(consKf);

  // fallback: 전체 대표 (sections가 있으면 섹션 기반 fallback 우선)
  if (!keys.length) {
    if (sectionFallbackKeys.length) keys.push(...sectionFallbackKeys);
    else keys.push("1_Ready_Total", "2_Rotation_Total", "3_Backswing_Total", "4_Impact_Total", "5_FollowSwing_Total");
  }

  renderYoutubeLinksByKfKeys(keys);
}

// LLM report -> 동작 카드(기존 actionCard UI)에 요약/피드백 반영
function renderActionCardsFromLLM(reportObj){
  const hasSections = reportObj && typeof reportObj === "object" && reportObj.sections && typeof reportObj.sections === "object";

  // New(score-based): sections -> 5 cards mapping
  const sectionMap = [
    { n: 1, skey: "ready", fallbackTitle: "준비" },
    { n: 2, skey: "rotation", fallbackTitle: "회전" },
    { n: 3, skey: "backswing", fallbackTitle: "백스윙" },
    { n: 4, skey: "impact", fallbackTitle: "임팩트" },
    { n: 5, skey: "followswing", fallbackTitle: "팔로스윙" },
  ];

  // Legacy: actions, only cards 3~5
  const actions = reportObj?.actions || {};
  const legacyMap = [
    { n: 1, key: null },
    { n: 2, key: null },
    { n: 3, key: "kf1" },
    { n: 4, key: "kf2" },
    { n: 5, key: "kf3" },
  ];

  for (let i = 0; i < 5; i++){
    const n = i + 1;

    // 1) pick content
    let title = null;
    let changeOne = null;
    let analysis = "-";

    if (hasSections){
      const s = reportObj?.sections?.[sectionMap[i].skey] || {};
      title = s?.title || sectionMap[i].fallbackTitle;
      changeOne = s?.change_one || "-";
      analysis = s?.analysis || "-";
    } else {
      const legacyKey = legacyMap[i].key;
      const a = legacyKey ? (actions?.[legacyKey] || {}) : {};
      title = a?.title ? String(a.title) : (legacyKey ? actionNameFromKfKey(legacyKey) : sectionMap[i].fallbackTitle);
      changeOne = a?.problem_one ? String(a.problem_one) : "-";
      analysis = Array.isArray(a?.fix_two) && a.fix_two.length
        ? a.fix_two.map((x)=>String(x)).join(" ")
        : (a?.problem_one ? String(a.problem_one) : "-");
    }

    // 2) inject/replace LLM block only
    const body = document.getElementById(`a${n}Body`);
    if (body){
      const existing = body.querySelector(".llmActionBlock");
      const html = `
        <div class="llmActionBlock" style="margin-top:10px; padding-top:10px; border-top:1px dashed rgba(17,24,39,.18);">
          <div style="margin-top:6px;">${String(analysis || "-")}</div>
        </div>
      `;

      if (existing){
        existing.outerHTML = html;
      } else {
        body.insertAdjacentHTML("beforeend", html);
      }
    }

    // 3) meta badge
    const meta = document.getElementById(`a${n}Meta`);
    if (meta){
      if (!meta.querySelector(".llmAppliedBadge")){
        meta.insertAdjacentHTML(
          "beforeend",
          ` <span class="llmAppliedBadge" style="margin-left:6px; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:900; background:rgba(32,201,151,.14); border:1px solid rgba(32,201,151,.28); color:rgba(17,24,39,.86);">SCORE 비교</span>`
        );
      }
    }
  }
}

function renderLLMReport(reportObj){
  const DEV = location.hostname === "localhost";
  if (DEV) {
    console.groupCollapsed("[LLM REPORT RAW JSON]");
    console.log(reportObj);
    console.groupEnd();
  }

  // 상단 성장 문구(summarySub)는 DB 비교(comparison) 기반으로 유지합니다.
  // LLM의 summary는 별도 영역이 있을 때만 표시합니다.
  const summaryText = reportObj?.summary ? String(reportObj.summary) : null;
  const growth = reportObj?.growth || null;
  const deltaAvg = growth && Number.isFinite(Number(growth.delta_average_score)) ? Number(growth.delta_average_score).toFixed(2) : null;
  const growthMsg = growth?.message ? String(growth.message) : null;

  const llmSumEl = document.getElementById("llmSummary");
  if (llmSumEl) {
    if (growthMsg && deltaAvg != null) llmSumEl.textContent = `${growthMsg} (Δ ${deltaAvg})`;
    else llmSumEl.textContent = summaryText || "-";
  }

  renderActionCardsFromLLM(reportObj);
  renderYoutubeLinksFromReport(reportObj);
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
async function init() {
  const payload = await loadFromDB(__RANGE_FILTER__);

  const current = Array.isArray(payload?.current_sessions) ? payload.current_sessions : [];
  const prev = Array.isArray(payload?.prev_sessions) ? payload.prev_sessions : [];
  const comp = payload?.comparison || null;

  const last = current.length ? current[current.length - 1] : null;

  // 세션 히스토리 전역 상태 (현재 기간 기준)
  __ALL_SESSIONS__ = current;

  // 상단 성장 문구 + 링 색상 (초기)
  renderGrowthSummary(comp, payload?.range || __RANGE_FILTER__);
  setScoreRingColor(comp?.direction);
  // 세션 히스토리 전역 상태 (현재 기간 기준)
  __ALL_SESSIONS__ = current;

  // 상단 성장 문구 + 링 색상 (초기)
  renderGrowthSummary(comp, payload?.range || __RANGE_FILTER__);
  setScoreRingColor(comp?.direction);

  // KF 탭 클릭 시 차트를 현재 필터 기준으로 재렌더
  wireRangeTabs(async (r) => {
    try {
      await refreshByRange(r);
    } catch (e) {
      alert(e.message);
    }
  });

  // 종합 리포트 페이지: 세션 히스토리(여러 번의 평가)를 시각화
  renderScoreKfHistoryChart(current, prev);

  // KF별 분석 차트
  renderActionCards(current, prev);
  // 동작 카드 캐러셀 스크롤 감지(에러 방지)
  wireActionCarousel();
  renderScoreKfHistoryChart(current, prev);

  // KF별 분석 차트
  renderActionCards(current, prev);
  // 동작 카드 캐러셀 스크롤 감지(에러 방지)
  wireActionCarousel();

  // 현재 세션(가장 최근) 스냅샷
  window.__CURRENT_FRAME__ = last?.frame ?? "ALL";
  const meta = last?.meta || {};

  // 메타/스냅샷 렌더
  renderMeta({ ...meta, created_at: last?.created_at, idx: last?.idx });

  // LLM 생성 버튼 wiring (LLM 섹션을 숨겨도 버튼은 유지)
  wireLLMGenerateButton();
  // LLM 생성 버튼 wiring (LLM 섹션을 숨겨도 버튼은 유지)
  wireLLMGenerateButton();

  // 초기 점수: 최근 세션의 score 또는 kf_error 기반
  const initialScore = Number.isFinite(Number(last?.score))
    ? Number(last.score)
    : computeScoreFromKfError(last?.kf_error);

  setScore(initialScore);

  // Render latest LLM report on first load if available
  const latestLLM = payload?.latest_llm_report?.report || null;
  if (latestLLM) {
    renderLLMReport(latestLLM);
  }
}

// ====== LLM 리포트 생성/갱신 버튼 ======
function wireLLMGenerateButton(){
  const btn = document.getElementById("btnGenerateLLM");
  if (!btn) return;

  btn.addEventListener("click", async ()=>{
    try{
      btn.disabled = true;
      btn.classList.add("is-active");
      btn.textContent = "생성 중...";
      console.log("[LLM GENERATE] post_idx=", (getPostIdxFromURL() || getPostIdxFallback()), "range=", __RANGE_FILTER__);
      const postIdx = getPostIdxFromURL() || getPostIdxFallback();
      if (!postIdx) throw new Error("post_idx가 없습니다. URL에 ?post_idx=... 를 붙이세요.");
      const report = await generateLLMReportByPostIdx(postIdx, "ko");
      renderLLMReport(report);

      // (선택) 생성 후 분석 데이터도 다시 로딩해서 차트/트렌드 갱신
      await refreshByRange(__RANGE_FILTER__);

      btn.textContent = "LLM 리포트 생성";
      btn.classList.remove("is-active");
      btn.disabled = false;
    }catch(e){
      btn.textContent = "LLM 리포트 생성";
      btn.classList.remove("is-active");
      btn.disabled = false;
      alert(e.message);
    }
  });
}