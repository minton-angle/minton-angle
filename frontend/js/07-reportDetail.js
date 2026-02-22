/**
 * 07-reportDetail.js
 * 실시간 3회 스윙 탭 전환 및 단일 분석 대응 통합 버전
 */

// 글로벌 상태 변수
let videoFPS = 30; 
let kf1_frame = 0; 
let kf3_frame = 0; 
let allSwingData = {}; // 🌟 실시간 3회분 데이터를 담을 객체
let currentType = 'video'; // 기본값

document.addEventListener('DOMContentLoaded', () => {
    loadAnalysisResult();
});

/**
 * 1. 백엔드로부터 분석 데이터 로드
 */
async function loadAnalysisResult() {
    const urlParams = new URLSearchParams(window.location.search);
    const postIdx = urlParams.get('post_id');
    currentType = urlParams.get('type') || 'video'; // type 파라미터 읽기
    
    if (!postIdx || postIdx === 'null') {
        showError('분석 결과를 찾을 수 없습니다.');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/report/upload/result/${postIdx}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const result = await response.json();
        console.log('✅ 서버로부터 받은 전체 데이터:', result);
        
        // 🌟 실시간 모드일 때 탭 처리
        if (currentType === 'realtime') {
            document.getElementById('realtime-tabs').style.display = 'flex';
            
            // 백엔드가 스윙별로 데이터를 묶어서 준다고 가정 (예: result.swings[1], result.swings[2]...)
            // 만약 구조가 다르면 이 부분을 백엔드 구조에 맞게 매핑하면 됩니다.
            allSwingData = result.swings || { 1: result }; 
            
            // 처음에는 1회차 결과 표시
            switchSwing(1);
        } else {
            // 일반 영상 업로드 모드
            if (result.success && result.scores) {
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

/**
 * 🌟 실시간 모드용 탭 전환 함수
 */
function switchSwing(swingNum) {
    // 1. 버튼 활성화 스타일 변경
    document.querySelectorAll('.tab-btn').forEach((btn, idx) => {
        btn.classList.toggle('active', (idx + 1) === swingNum);
    });

    // 2. 해당 회차 데이터가 있는지 확인 후 화면 갱신
    const swingData = allSwingData[swingNum];
    if (swingData) {
        // 키프레임 프레임 번호 갱신 (Phase 2 제어용)
        kf1_frame = swingData.kf1 || 0;
        kf3_frame = swingData.kf3 || 100;
        displayResult(swingData);
        console.log(`✅ 스윙 ${swingNum}회차 데이터로 전환 완료`);
    } else {
        console.warn(`⚠️ 스윙 ${swingNum}회차 데이터가 존재하지 않습니다.`);
    }
}

/**
 * 2. 화면 데이터 바인딩 (공용)
 */
function displayResult(result) {
    const scoreData = result.scores || result; // 데이터 구조 유연성 확보
    const files = result.files;      

    displayOverallScore(result.total_score || scoreData.total_score || 0);
    
    if (scoreData.stage_scores) {
        displayStageScores(scoreData.stage_scores);
    }
    
    if (scoreData.evaluation) {
        displayEvaluation(scoreData.evaluation);
    }
    
    if (files) {
        displayMediaComparison(files);
    }
}

/**
 * 3. 종합 점수 및 코멘트 표시
 */
function displayOverallScore(score) {
    const scoreVal = Math.round(score);
    document.getElementById('overall-score').textContent = scoreVal;
    
    const gradeEl = document.getElementById('overall-grade');
    const commentEl = document.getElementById('overall-comment');
    
    let gradeText, comment;
    if (scoreVal >= 80) { gradeText = '완벽해요! 🎉'; comment = '전문가 수준의 스윙입니다. 폼이 아주 훌륭해요!'; }
    else if (scoreVal >= 60) { gradeText = '잘하고 있어요! 👍'; comment = '기본이 탄탄합니다. 몇 가지만 개선하면 완벽해질 거예요.'; }
    else if (scoreVal >= 40) { gradeText = '조금 더 연습해봐요 💪'; comment = '아래 피드백을 참고해서 스윙 궤적을 교정해보세요.'; }
    else { gradeText = '기초부터 다시! 📚'; comment = '정확한 타점과 팔 펴짐 동작에 집중해 연습하세요.'; }
    
    gradeEl.textContent = gradeText;
    commentEl.textContent = comment;
    
    const meter = document.getElementById('score-meter');
    if (meter) {
        const circumference = 2 * Math.PI * 45;
        const offset = circumference - (scoreVal / 100) * circumference;
        meter.style.strokeDasharray = `${circumference} ${circumference}`;
        meter.style.strokeDashoffset = offset;
    }
}

/**
 * 4. 단계별 점수 배지 업데이트
 */
function displayStageScores(stageScores) {
    const s1 = document.getElementById('phase1-badge');
    const s2 = document.getElementById('phase2-badge');
    const s3 = document.getElementById('phase3-badge');

    if(s1) { s1.textContent = `${stageScores.stage1 || 0}점`; s1.className = 'phase-badge ' + getScoreClass(stageScores.stage1); }
    if(s2) { s2.textContent = `${stageScores.stage2 || 0}점`; s2.className = 'phase-badge ' + getScoreClass(stageScores.stage2); }
    if(s3) { s3.textContent = `${stageScores.stage3 || 0}점`; s3.className = 'phase-badge ' + getScoreClass(stageScores.stage3); }
}

function getScoreClass(score) {
    if (score >= 80) return 'high';
    if (score >= 50) return 'mid';
    return 'low';
}

/**
 * 5. 상세 지표 PASS/FAIL 업데이트
 */
function displayEvaluation(evaluation) {
    // 기존에 적용된 pass/fail 클래스들 초기화
    document.querySelectorAll('.eval-item').forEach(el => el.classList.remove('pass', 'fail'));

    evaluation.forEach(item => {
        const statusEl = document.getElementById(`eval-${item.id}`);
        if (!statusEl) return;

        const itemEl = statusEl.closest('.eval-item');
        
        if (item.pass === 1) {
            statusEl.textContent = 'PASS ✓';
            statusEl.className = 'eval-status pass';
            itemEl?.classList.add('pass');
        } else {
            statusEl.textContent = item.status || 'FAIL';
            statusEl.className = 'eval-status fail';
            itemEl?.classList.add('fail');
        }
    });
}

/**
 * 6. 미디어 파일 경로 설정 및 영상 초기화
 */
function displayMediaComparison(files) {
    const fixPath = (rawPath) => {
        if (!rawPath) return "";
        let cleanPath = rawPath.replace(/\\/g, '/');
        const marker = "backend/data/";
        const index = cleanPath.indexOf(marker);
        if (index !== -1) {
            return "/" + cleanPath.substring(index);
        }
        return cleanPath;
    };

    // 1단계 준비자세
    if (files.kf1_image) {
        document.getElementById('phase1-user-img').src = `${API_BASE_URL}${fixPath(files.kf1_image)}`;
    }
    
    // 2단계 스윙 영상
    const v2User = document.getElementById('phase2-user-video');
    const v2Expert = document.getElementById('phase2-expert-video');

    if (files.backswing_video) {
        v2User.src = `${API_BASE_URL}${fixPath(files.backswing_video)}`;
        v2Expert.src = "assets/2_rotation_hybrid.mp4"; 
        
        v2User.onloadedmetadata = () => { syncToFrame(kf1_frame); };
        v2Expert.onloadedmetadata = () => { syncToFrame(kf1_frame); };
        v2User.load();
        v2Expert.load();
    }

    // 2단계 고정 키프레임
    if (files.kf2_image) {
        document.getElementById('phase2-backswing-img').src = `${API_BASE_URL}${fixPath(files.kf2_image)}`;
    }
    if (files.kf3_image) {
        document.getElementById('phase2-impact-img').src = `${API_BASE_URL}${fixPath(files.kf3_image)}`;
    }

    // 3단계 팔로우스루
    if (files.impact_video) {
        const v3 = document.getElementById('phase3-user-video');
        v3.src = `${API_BASE_URL}${fixPath(files.impact_video)}`;
        v3.load();
    }
}

/**
 * 7. 프레임 단위 이동 제어 함수
 */
function stepFrame(offset) {
    const v2User = document.getElementById('phase2-user-video');
    if (!v2User) return;

    let currentFrame = Math.round(v2User.currentTime * videoFPS);
    let targetFrame = currentFrame + offset;

    if (targetFrame < kf1_frame) targetFrame = kf1_frame;
    if (targetFrame > kf3_frame) targetFrame = kf3_frame;

    syncToFrame(targetFrame);
}

function syncToFrame(frameNumber) {
    const v2User = document.getElementById('phase2-user-video');
    const v2Expert = document.getElementById('phase2-expert-video');
    if (!v2User || !v2Expert) return;
    
    const targetTime = frameNumber / videoFPS;
    v2User.currentTime = targetTime;
    v2Expert.currentTime = targetTime;

    const label = document.getElementById('current-step-label');
    if (label) {
        const totalSection = kf3_frame - kf1_frame;
        const currentPos = frameNumber - kf1_frame;
        const progress = totalSection > 0 ? Math.round((currentPos / totalSection) * 100) : 0;
        label.textContent = `스윙 분석 (${progress}%)`;
    }
}

function syncPlayVideos(phase) {
    const userVideo = document.getElementById(`${phase}-user-video`);
    const expertVideo = document.getElementById(`${phase}-expert-video`);
    if (userVideo && expertVideo) {
        userVideo.currentTime = 0;
        expertVideo.currentTime = 0;
        userVideo.play();
        expertVideo.play();
    }
}

/**
 * 9. 에러 발생 시 UI 처리
 */
function showError(message) {
    const content = document.querySelector('.report-content');
    if (!content) return;
    content.innerHTML = `
        <div style="padding: 60px 20px; text-align: center; background: #fff; border-radius: 20px; margin: 20px;">
            <p style="font-size: 18px; color: #666; margin-bottom: 24px; word-break: keep-all;">😢 ${message}</p>
            <button onclick="location.href='01-home.html'" style="padding: 14px 28px; background: #025B36; color: #fff; border: none; border-radius: 12px; font-weight: 700; cursor: pointer;">홈으로 돌아가기</button>
        </div>
    `;
}

console.log('📄 07-reportDetail.js 로드 완료');