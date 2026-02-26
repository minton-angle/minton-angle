/**
 * 07-reportDetail.js
 * [최종 통합본] 팀원 업데이트(실시간 탭/슬라이더) + 배포 환경(ngrok/Docker) 대응
 */

let currentType = 'video';
let allSwingData = {};
let currentSeqIdx = 0;
let seqFiles = {};

const SEQ_KEYS = [
    'seq1_ready', 'seq2_takeaway', 'seq3_backswing',
    'seq4_downswing1', 'seq5_downswing2', 'seq6_impact'
];
const SEQ_LABELS = ['준비', '테이크어웨이', '백스윙', '다운스윙1', '다운스윙2', '임팩트'];

// 전문가 시퀀스 이미지 경로 (로컬 에셋)
const EXPERT_SEQ = [
    'assets/Seq_1_Ready.jpg',
    'assets/Seq_2_Takeaway.jpg',
    'assets/Seq_3_Backswing.jpg',
    'assets/Seq_4_Downswing_1.jpg',
    'assets/Seq_5_Downswing_2.jpg',
    'assets/Seq_6_Impact.jpg',
];

document.addEventListener('DOMContentLoaded', () => {
    loadAnalysisResult();
});

// ------------------------------------------------------------------ //
//  1. 배포 환경 경로 치환 유틸리티
// ------------------------------------------------------------------ //
function fixPath(raw) {
    if (!raw) return '';
    let p = raw.replace(/\\/g, '/');
    if (p.startsWith('http')) return p;
    // Docker 내부 경로(/app/data)를 외부 접근 경로(/data)로 변환
    if (p.includes('/app/data')) p = p.replace('/app/data', '/data');
    if (p.includes('/data/data')) p = p.replace('/data/data', '/data');
    
    const marker = 'backend/data/';
    const idx = p.indexOf(marker);
    if (idx !== -1) p = '/' + p.substring(idx);
    
    return p.startsWith('/') ? p : '/' + p;
}

// ------------------------------------------------------------------ //
//  2. 백엔드 데이터 로드 (apiCall/ngrok 대응)
// ------------------------------------------------------------------ //
async function loadAnalysisResult() {
    const urlParams = new URLSearchParams(window.location.search);
    const postIdx = urlParams.get('post_id');
    currentType = urlParams.get('type') || 'video';

    if (!postIdx || postIdx === 'null') {
        showError('분석 결과를 찾을 수 없습니다.');
        return;
    }

    try {
        // API 엔드포인트 설정
        const endpoint = currentType === 'realtime' 
            ? `/api/report/realtime/result/${postIdx}` 
            : `/api/upload/result/${postIdx}`;

        console.log(`📡 데이터 요청 중... (${currentType})`);

        // common.js의 apiCall 사용 (ngrok 헤더 및 인증 자동 처리)
        const result = await apiCall(endpoint, {
            method: 'GET',
            auth: true,
            headers: { "ngrok-skip-browser-warning": "true" } // ngrok 경고 페이지 우회
        });

        console.log('✅ 서버 응답 전체:', result);

        if (currentType === 'realtime' && result.swings) {
            // 실시간 모드일 때 탭 표시
            document.getElementById('realtime-tabs').style.display = 'flex';
            allSwingData = result.swings;
            switchSwing(1); // 1회차 데이터 먼저 로드
        } else {
            // 일반 업로드 모드
            if (result.success || result.scores) {
                displayResult(result);
            } else {
                throw new Error('결과 데이터 형식이 올바르지 않습니다.');
            }
        }

    } catch (e) {
        console.error('❌ API 오류:', e);
        showError('분석 결과를 불러오는 중 오류가 발생했습니다.');
    }
}

// ------------------------------------------------------------------ //
//  3. 실시간 스윙 회차 전환 (팀원 개선 로직 통합)
// ------------------------------------------------------------------ //
function switchSwing(swingNum) {
    document.querySelectorAll('.tab-btn').forEach((btn, idx) => {
        btn.classList.toggle('active', (idx + 1) === swingNum);
    });

    const swingData = allSwingData[swingNum] || allSwingData[String(swingNum)];
    if (!swingData) {
        console.warn(`⚠️ ${swingNum}회차 데이터 없음`, allSwingData);
        return;
    }

    // ⭐ 슬라이더 파일 정보 갱신 (참조 끊기 위해 복사)
    currentSeqIdx = 0;
    seqFiles = { ...swingData.files }; 
    
    console.log(`🔄 ${swingNum}회차 전환:`, seqFiles.seq1_ready);

    displayResult({
        success: true,
        total_score: swingData.total_score,
        scores: swingData.scores,
        files: swingData.files
    });

    // 슬라이더 강제 업데이트
    updateSeqSlider();
}

// ------------------------------------------------------------------ //
//  4. 전체 화면 바인딩
// ------------------------------------------------------------------ //
function displayResult(result) {
    let data = result;
    
    // 실시간 모드 데이터 구조 조정
    if (result.type === 'realtime' && result.swings) {
        const swing3 = result.swings['3'] || result.swings[3];
        if (swing3) {
            data = {
                ...result,
                total_score: swing3.total_score,
                scores: swing3.scores,
                files: swing3.files
            };
        }
    }

    displayOverallScore(data.total_score || 0);

    const details = data.scores?.details || data.details || {};
    displayStageScores(details);
    displayEvaluation(details);

    if (data.files) {
        displayMediaComparison(data.files);
        initSeqSlider(data.files);
    }
}

// ------------------------------------------------------------------ //
//  5. 시각화 및 미디어 처리 (배포 경로 적용)
// ------------------------------------------------------------------ //
function displayOverallScore(score) {
    const scoreVal = Math.round(score);
    document.getElementById('overall-score').textContent = scoreVal;

    const gradeEl = document.getElementById('overall-grade');
    const commentEl = document.getElementById('overall-comment');

    let gradeText, comment;
    if (scoreVal >= 80)      { gradeText = '완벽해요! 🎉';          comment = '전문가 수준의 스윙입니다. 폼이 아주 훌륭해요!'; }
    else if (scoreVal >= 60) { gradeText = '잘하고 있어요! 👍';      comment = '기본이 탄탄합니다. 몇 가지만 개선하면 완벽해질 거예요.'; }
    else if (scoreVal >= 40) { gradeText = '조금 더 연습해봐요 💪'; comment = '아래 피드백을 참고해서 스윙 궤적을 교정해보세요.'; }
    else                     { gradeText = '기초부터 다시! 📚';      comment = '정확한 타점과 팔 펴짐 동작에 집중해 연습하세요.'; }

    gradeEl.textContent = gradeText;
    commentEl.textContent = comment;

    const meter = document.getElementById('score-meter');
    if (meter) {
        const circumference = 2 * Math.PI * 45;
        meter.style.strokeDasharray = `${circumference} ${circumference}`;
        meter.style.strokeDashoffset = circumference - (scoreVal / 100) * circumference;
    }
}

function displayMediaComparison(files) {
    const setImg = async (id, path) => {
        const el = document.getElementById(id);
        if (el && path) {
            const baseUrl = API_BASE_URL.endsWith('/') ? API_BASE_URL.slice(0, -1) : API_BASE_URL;
            const finalUrl = `${baseUrl}${fixPath(path)}`;

            try {
                // 🌟 핵심: ngrok 경고 페이지를 무시하는 헤더를 들고 사진을 직접 요청합니다.
                const response = await fetch(finalUrl, {
                    headers: {
                        'ngrok-skip-browser-warning': '69420'
                    }
                });
                
                // 받아온 데이터를 이미지로 변환
                const blob = await response.blob();
                const objectURL = URL.createObjectURL(blob);
                el.src = objectURL;
                console.log(`✅ ${id} 이미지 직빵 로드 성공!`);
            } catch (error) {
                console.error(`❌ 이미지 로드 실패: ${id}`, error);
            }
        }
    };

    // Phase 1 & 2 이미지
    setImg('phase1-user-img', files.ready || files.kf1_image);
    setImg('phase2-backswing-img', files.seq3_backswing);
    setImg('phase2-impact-img',    files.seq6_impact);

    // Phase 3 영상
    setVideo('phase3-user-video', files.followswing || files.follow_video);
}

async function setVideo(id, path) {
    const el = document.getElementById(id);
    if (el && path) {
        const baseUrl = API_BASE_URL.endsWith('/') ? API_BASE_URL.slice(0, -1) : API_BASE_URL;
        const finalUrl = `${baseUrl}${fixPath(path)}`;
        try {
            const response = await fetch(finalUrl, {
                headers: { 'ngrok-skip-browser-warning': '69420' }
            });
            const blob = await response.blob();
            el.src = URL.createObjectURL(blob);
            el.load();
            console.log(`✅ ${id} 영상 로드 성공`);
        } catch (e) {
            console.error(`❌ 영상 로드 실패: ${id}`, e);
        }
    }
}

// ------------------------------------------------------------------ //
//  6. 슬라이더 로직
// ------------------------------------------------------------------ //
function initSeqSlider(files) {
    seqFiles = files;
    currentSeqIdx = 0;
    updateSeqSlider();
}

function stepSeq(dir) {
    currentSeqIdx = (currentSeqIdx + dir + 6) % 6;
    updateSeqSlider();
}

async function updateSeqSlider() {
    // 내 스윙 이미지 업데이트
    const userImg = document.getElementById('phase2-user-seq-img');
    const userPath = seqFiles[SEQ_KEYS[currentSeqIdx]];
    if (userImg && userPath) {
        const fullUrl = `${API_BASE_URL}${fixPath(userPath)}`;
        try {
            const response = await fetch(fullUrl, {
                headers: { 'ngrok-skip-browser-warning': '69420' }
            });
            const blob = await response.blob();
            userImg.src = URL.createObjectURL(blob);
        } catch (e) {
            console.error('슬라이더 이미지 로드 실패', e);
        }
    }

    // 전문가 이미지는 로컬 assets이라 그대로 둬도 됨
    const expertImg = document.getElementById('phase2-expert-seq-img');
    if (expertImg) expertImg.src = EXPERT_SEQ[currentSeqIdx];

    const label = document.getElementById('current-step-label');
    if (label) label.textContent = `${SEQ_LABELS[currentSeqIdx]} (${currentSeqIdx + 1}/6)`;
}

// ------------------------------------------------------------------ //
//  7. 점수 배지 및 평가 결과
// ------------------------------------------------------------------ //
function displayStageScores(details) {
    const readyScores = Object.values(details.Ready || {}).map(v => v.score || 0);
    const s1 = readyScores.length ? Math.round(readyScores.reduce((a,b) => a+b, 0) / readyScores.length) : 0;

    const rotScores = Object.values(details.Rotation || {}).map(v => v.score || 0);
    const bsScores  = Object.values(details.Backswing || {}).map(v => v.score || 0);
    const swingAll  = [...rotScores, ...bsScores];
    const s2 = swingAll.length ? Math.round(swingAll.reduce((a,b) => a+b, 0) / swingAll.length) : 0;

    const impScores = Object.values(details.Impact || {}).map(v => v.score || 0);
    const fwScore   = details.FollowSwing?.Performance?.score || 0;
    const phase3All = [...impScores, fwScore];
    const s3 = phase3All.length ? Math.round(phase3All.reduce((a,b) => a+b, 0) / phase3All.length) : 0;

    const b1 = document.getElementById('phase1-badge');
    const b2 = document.getElementById('phase2-badge');
    const b3 = document.getElementById('phase3-badge');

    if (b1) { b1.textContent = `${s1}점`; b1.className = 'phase-badge ' + getScoreClass(s1); }
    if (b2) { b2.textContent = `${s2}점`; b2.className = 'phase-badge ' + getScoreClass(s2); }
    if (b3) { b3.textContent = `${s3}점`; b3.className = 'phase-badge ' + getScoreClass(s3); }
}

function getScoreClass(score) {
    if (score >= 80) return 'high';
    if (score >= 50) return 'mid';
    return 'low';
}

function displayEvaluation(details) {
    const ready = details.Ready || {};
    setEval('eval-ready-arm', ready.Arm_Angle, v => v >= 18 && v <= 70 ? '적정' : v > 70 ? '넓음' : '좁음', d => d.measured);
    setEval('eval-ready-wrist-h', ready.Left_Wrist_Height, v => v < 0 ? '적정' : v < 0.1 ? '낮음' : '높음', d => d.measured);
    setEval('eval-ready-stance', ready.Stance_Width, v => v >= 100 ? '적정' : '좁음', d => d.score);
    setEval('eval-ready-ratio', ready.Wrist_Height_Ratio, v => v >= -0.5 && v <= 2.0 ? '적정' : v > 2.0 ? '높음' : '낮음', d => d.measured);

    const rot = details.Rotation || {};
    setEval('eval-rot-hip', rot.Hip_Max_Rotation, v => v >= 80 ? '안정' : '기울어짐', d => d.score);
    setEval('eval-rot-shoulder', rot.Shoulder_Max_Rotation, v => v >= 80 ? '적정' : v >= 60 ? '과함' : '부족', d => d.score);

    const bs = details.Backswing || {};
    setEval('eval-bs-wrist', bs.Wrist_X_Depth, v => v < -0.05 ? '깊음' : v < 0 ? '적정' : '얕음', d => d.measured);
    setEval('eval-bs-elbow', bs.Elbow_Lift, v => v >= 1.5 && v <= 3.0 ? '적정' : v > 3.0 ? '높음' : '낮음', d => d.measured);
    setEval('eval-bs-lshape', bs.L_Shape_Angle, v => v >= 60 && v <= 110 ? '적정' : v > 110 ? '넓음' : '좁음', d => d.measured);

    const imp = details.Impact || {};
    setEval('eval-impact-arm', imp.Arm_Extension_Angle, v => v >= 160 ? '펴짐' : v >= 140 ? '적정' : '굽힘', d => d.measured);
    setEval('eval-impact-wrist', imp.Wrist_Height_Ratio, v => v > 4.5 ? '높음' : '낮음', d => d.measured);

    const fw = details.FollowSwing?.Performance;
    const fwEl = document.getElementById('eval-follow');
    if (fwEl && fw) {
        const isPass = fw.score >= 100;
        const isMid  = fw.score >= 50;
        const label  = isPass ? '성공 ✓' : isMid ? '미흡' : '안함';
        activateOption(fwEl, label, isPass);
    }
}

function setEval(elId, data, labelFn, valFn) {
    const el = document.getElementById(elId);
    if (!el || !data) return;
    const val = valFn(data);
    const label = labelFn(val);
    const isPass = ['적정', '안정', '성공 ✓', '펴짐','높음'].includes(label);
    activateOption(el, label, isPass);
}

function activateOption(container, label, isPass) {
    container.querySelectorAll('span[data-val]').forEach(span => {
        span.classList.remove('active-pass', 'active-fail');
        if (span.dataset.val === label) span.classList.add(isPass ? 'active-pass' : 'active-fail');
    });
    const evalItem = container.closest('.eval-item');
    if (evalItem) {
        evalItem.classList.remove('pass', 'fail');
        evalItem.classList.add(isPass ? 'pass' : 'fail');
    }
}

// 동영상 동시 재생
function syncPlayVideos() {
    const videos = document.querySelectorAll('video');
    videos.forEach(v => {
        v.currentTime = 0;
        v.play();
    });
}

function showError(message) {
    const content = document.querySelector('.report-content');
    if (!content) return;
    content.innerHTML = `
        <div style="padding:60px 20px;text-align:center;background:#fff;border-radius:20px;margin:20px;">
            <p style="font-size:18px;color:#666;margin-bottom:24px;word-break:keep-all;">😢 ${message}</p>
            <button onclick="location.href='01-home.html'"
                style="padding:14px 28px;background:#025B36;color:#fff;border:none;border-radius:12px;font-weight:700;cursor:pointer;">
                홈으로 돌아가기
            </button>
        </div>`;
}

console.log('📄 07-reportDetail.js 배포 최적화 버전 로드 완료');