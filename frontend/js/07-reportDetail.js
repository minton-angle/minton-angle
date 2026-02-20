/**
 * 07-reportDetail.js
 * 전 단계 사용자-전문가 1:1 비교 및 Phase 2 수동 프레임 제어 통합 버전
 */

// 영상 제어를 위한 글로벌 변수
let videoFPS = 30; 
let kf1_frame = 0; // 준비 자세 프레임
let kf3_frame = 0; // 임팩트 자세 프레임 (최대 탐색 범위)

document.addEventListener('DOMContentLoaded', () => {
    loadAnalysisResult();
});

/**
 * 1. 백엔드로부터 분석 데이터 로드
 */
async function loadAnalysisResult() {
    const urlParams = new URLSearchParams(window.location.search);
    const postIdx = urlParams.get('post_id');
    
    if (!postIdx) {
        showError('분석 결과를 찾을 수 없습니다.');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/upload/result/${postIdx}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const result = await response.json();
        console.log('✅ 분석 데이터 수신:', result);
        
        if (result.success && result.scores) {
            // 키프레임 번호 저장 (Phase 2 제어용)
            kf1_frame = result.kf1 || 0;
            kf3_frame = result.kf3 || 100;
            displayResult(result);
        } else {
            throw new Error('결과 데이터 형식이 올바르지 않거나 존재하지 않습니다.');
        }
        
    } catch (e) {
        console.error('❌ API 오류:', e);
        showError('분석 결과를 불러오는 중 오류가 발생했습니다.');
    }
}

/**
 * 2. 화면 전체 데이터 바인딩
 */
function displayResult(result) {
    const scoreData = result.scores; 
    const files = result.files;      

    displayOverallScore(result.total_score || scoreData.total_score);
    
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
    document.getElementById('overall-score').textContent = score;
    
    const gradeEl = document.getElementById('overall-grade');
    const commentEl = document.getElementById('overall-comment');
    
    let gradeText, comment;
    if (score >= 90) { gradeText = '완벽해요! 🎉'; comment = '전문가 수준의 스윙입니다. 폼이 아주 훌륭해요!'; }
    else if (score >= 70) { gradeText = '잘하고 있어요! 👍'; comment = '기본이 탄탄합니다. 몇 가지만 개선하면 완벽해질 거예요.'; }
    else if (score >= 50) { gradeText = '조금 더 연습해봐요 💪'; comment = '아래 피드백을 참고해서 스윙 궤적을 교정해보세요.'; }
    else { gradeText = '기초부터 다시! 📚'; comment = '정확한 타점과 팔 펴짐 동작에 집중해 연습하세요.'; }
    
    gradeEl.textContent = gradeText;
    commentEl.textContent = comment;
    
    const meter = document.getElementById('score-meter');
    if (meter) {
        const circumference = 2 * Math.PI * 45;
        const offset = circumference - (score / 100) * circumference;
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

    if(s1) { s1.textContent = `${stageScores.stage1}점`; s1.className = 'phase-badge ' + getScoreClass(stageScores.stage1); }
    if(s2) { s2.textContent = `${stageScores.stage2}점`; s2.className = 'phase-badge ' + getScoreClass(stageScores.stage2); }
    if(s3) { s3.textContent = `${stageScores.stage3}점`; s3.className = 'phase-badge ' + getScoreClass(stageScores.stage3); }
}

function getScoreClass(score) {
    if (score >= 80) return 'high';
    if (score >= 50) return 'mid';
    return 'low';
}

/**
 * 5. 9개 상세 지표 PASS/FAIL 업데이트
 */
function displayEvaluation(evaluation) {
    evaluation.forEach(item => {
        const statusEl = document.getElementById(`eval-${item.id}`);
        const itemEl = statusEl?.closest('.eval-item');
        
        if (!statusEl) return;
        
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
    
    // 2단계 스윙 영상 (수동 제어 대상)
    const v2User = document.getElementById('phase2-user-video');
    const v2Expert = document.getElementById('phase2-expert-video');

    if (files.backswing_video) {
        v2User.src = `${API_BASE_URL}${fixPath(files.backswing_video)}`;
        v2Expert.src = "assets/2_rotation_hybrid.mp4"; // 전문가 기준 영상 경로
        
        // 메타데이터 로드 후 시작 프레임(kf1) 시점으로 자동 이동
        v2User.onloadedmetadata = () => { syncToFrame(kf1_frame); };
        v2Expert.onloadedmetadata = () => { syncToFrame(kf1_frame); };
    }

    // 2단계 고정 키프레임
    if (files.kf2_image) {
        document.getElementById('phase2-backswing-img').src = `${API_BASE_URL}${fixPath(files.kf2_image)}`;
    }
    if (files.kf3_image) {
        document.getElementById('phase2-impact-img').src = `${API_BASE_URL}${fixPath(files.kf3_image)}`;
    }

    // 3단계 팔로우스루 (기존 방식 유지)
    if (files.impact_video) {
        const v3 = document.getElementById('phase3-user-video');
        v3.src = `${API_BASE_URL}${fixPath(files.impact_video)}`;
        v3.load();
    }
}

/**
 * 7. [신규 추가] 프레임 단위 이동 제어 함수
 * @param {number} offset - 이동할 프레임 수 (5 또는 -5)
 */
function stepFrame(offset) {
    const v2User = document.getElementById('phase2-user-video');
    const v2Expert = document.getElementById('phase2-expert-video');
    
    if (!v2User || !v2Expert) return;

    // 현재 시간을 프레임 번호로 변환
    let currentFrame = Math.round(v2User.currentTime * videoFPS);
    let targetFrame = currentFrame + offset;

    // 탐색 범위 제한 (kf1_frame ~ kf3_frame)
    if (targetFrame < kf1_frame) targetFrame = kf1_frame;
    if (targetFrame > kf3_frame) targetFrame = kf3_frame;

    syncToFrame(targetFrame);
}

/**
 * 특정 프레임 번호로 두 영상을 동기화하여 이동
 */
function syncToFrame(frameNumber) {
    const v2User = document.getElementById('phase2-user-video');
    const v2Expert = document.getElementById('phase2-expert-video');
    
    const targetTime = frameNumber / videoFPS;
    
    v2User.currentTime = targetTime;
    v2Expert.currentTime = targetTime;

    // UI 레이블 업데이트
    const label = document.getElementById('current-step-label');
    if (label) {
        const totalSection = kf3_frame - kf1_frame;
        const currentPos = frameNumber - kf1_frame;
        const progress = totalSection > 0 ? Math.round((currentPos / totalSection) * 100) : 0;
        label.textContent = `스윙 분석 (${progress}%)`;
    }
}

/**
 * 8. 영상 동시 재생 제어 (Phase 3 전용)
 */
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
message.txt