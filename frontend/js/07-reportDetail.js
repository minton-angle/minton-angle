/**
 * 07-reportDetail.js
 * 스윙 분석 리포트 상세 페이지
 */

// ============================================
// 설정
// ============================================
const API_BASE_URL = 'http://localhost:8000/api';

// ============================================
// 초기화
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    loadAnalysisResult();
});

// ============================================
// 분석 결과 로드
// ============================================
async function loadAnalysisResult() {
    // localStorage에서 analysis_id 가져오기
    const analysisId = localStorage.getItem('analysis_id');
    const analysisData = localStorage.getItem('analysis_result');
    
    if (analysisData) {
        // localStorage에 결과가 있으면 바로 사용
        try {
            const result = JSON.parse(analysisData);
            displayResult(result);
        } catch (e) {
            console.error('결과 파싱 오류:', e);
            showError('결과를 불러올 수 없습니다.');
        }
    } else if (analysisId) {
        // analysis_id로 서버에서 결과 조회
        try {
            const response = await fetch(`${API_BASE_URL}/swing/result/${analysisId}`);
            if (!response.ok) throw new Error('결과 조회 실패');
            const result = await response.json();
            displayResult(result);
        } catch (e) {
            console.error('API 호출 오류:', e);
            // 테스트용 더미 데이터 표시
            displayDummyResult();
        }
    } else {
        // 테스트용 더미 데이터 표시
        displayDummyResult();
    }
}

// ============================================
// 결과 표시
// ============================================
function displayResult(result) {
    if (!result.success) {
        showError(result.message || '분석에 실패했습니다.');
        return;
    }
    
    // 종합 점수
    displayOverallScore(result.overall);
    
    // 피드백
    displayFeedback(result.overall.feedback_summary);
    
    // 구간별 결과
    displayPhase1(result.phases.phase1_ready, result.analysis_id);
    displayPhase2(result.phases.phase2_backswing_impact, result.analysis_id);
    displayPhase3(result.phases.phase3_followthrough, result.analysis_id);
}

// ============================================
// 종합 점수 표시
// ============================================
function displayOverallScore(overall) {
    const scoreEl = document.getElementById('overall-score');
    const gradeEl = document.getElementById('overall-grade');
    const commentEl = document.getElementById('overall-comment');
    const circleEl = document.getElementById('overall-score-circle');
    
    scoreEl.textContent = overall.score;
    
    // 등급별 텍스트 & 스타일
    const gradeInfo = {
        'excellent': { text: '완벽해요! 🎉', comment: '전문가 수준의 스윙이에요!' },
        'good': { text: '좋아요! 👍', comment: '조금만 더 연습하면 완벽해질 거예요.' },
        'fair': { text: '보통이에요', comment: '몇 가지 개선점을 확인해보세요.' },
        'poor': { text: '연습이 필요해요', comment: '아래 피드백을 참고해서 연습해보세요.' }
    };
    
    const grade = overall.grade || 'fair';
    const info = gradeInfo[grade] || gradeInfo.fair;
    
    gradeEl.textContent = info.text;
    commentEl.textContent = info.comment;
    circleEl.className = `score-circle ${grade}`;
}

// ============================================
// 피드백 표시
// ============================================
function displayFeedback(feedbackList) {
    const listEl = document.getElementById('feedback-list');
    listEl.innerHTML = '';
    
    if (!feedbackList || feedbackList.length === 0) {
        feedbackList = ['전반적으로 좋은 스윙이에요!'];
    }
    
    feedbackList.forEach(feedback => {
        const li = document.createElement('li');
        li.textContent = feedback.replace(/^\[.*?\]\s*/, ''); // [태그] 제거
        listEl.appendChild(li);
    });
}

// ============================================
// 구간1: 준비자세
// ============================================
function displayPhase1(phase, analysisId) {
    if (!phase) return;
    
    const imgEl = document.getElementById('phase1-user-img');
    
    if (phase.file_url) {
        imgEl.src = `${API_BASE_URL}${phase.file_url}`;
    }
}

// ============================================
// 구간2: 백스윙~임팩트
// ============================================
function displayPhase2(phase, analysisId) {
    if (!phase) return;
    
    // 영상 설정
    const videoEl = document.getElementById('phase2-user-video');
    if (phase.file_url) {
        videoEl.querySelector('source').src = `${API_BASE_URL}${phase.file_url}`;
        videoEl.load();
    }
    
    // 임팩트 이미지
    const impactImg = document.getElementById('phase2-impact-img');
    if (phase.impact_image_url) {
        impactImg.src = `${API_BASE_URL}${phase.impact_image_url}`;
    }
    
    // 지표 표시
    if (phase.metrics) {
        displayMetric('elbow', phase.metrics.elbow_angle);
        displayMetric('height', phase.metrics.impact_height);
        displayMetric('hip', phase.metrics.hip_rotation);
        
        // 구간 점수 (지표 평균)
        const avgScore = calculatePhaseScore(phase.metrics);
        document.getElementById('phase2-badge').textContent = `${avgScore}점`;
    }
}

// ============================================
// 구간3: 팔로우스루
// ============================================
function displayPhase3(phase, analysisId) {
    if (!phase) return;
    
    // 영상 설정
    const videoEl = document.getElementById('phase3-user-video');
    if (phase.file_url) {
        videoEl.querySelector('source').src = `${API_BASE_URL}${phase.file_url}`;
        videoEl.load();
    }
    
    // 지표 표시
    if (phase.metrics) {
        displayMetric('followthrough', phase.metrics.followthrough);
        
        // 구간 점수
        const score = phase.metrics.followthrough?.score || 0;
        document.getElementById('phase3-badge').textContent = `${score}점`;
    }
}

// ============================================
// 개별 지표 표시
// ============================================
function displayMetric(metricKey, metricData) {
    if (!metricData) return;
    
    const valueEl = document.getElementById(`${metricKey}-value`);
    const barEl = document.getElementById(`${metricKey}-bar`);
    const feedbackEl = document.getElementById(`${metricKey}-feedback`);
    const itemEl = document.getElementById(`metric-${metricKey}`);
    
    // 값 표시
    let displayValue = metricData.value;
    if (metricData.unit === '°') {
        displayValue = `${Math.round(metricData.value)}°`;
    } else if (metricData.unit === '%') {
        displayValue = `${Math.round(metricData.value)}%`;
    } else {
        displayValue = metricData.value.toFixed(2);
    }
    valueEl.textContent = displayValue;
    
    // 피드백
    feedbackEl.textContent = metricData.feedback || '';
    
    // 등급별 스타일
    itemEl.classList.remove('good', 'fair', 'poor');
    itemEl.classList.add(metricData.grade || 'fair');
    
    // 바 위치 조정 (0~100% 범위)
    const position = calculateBarPosition(metricKey, metricData.value);
    barEl.style.setProperty('--indicator-position', `${position}%`);
    barEl.querySelector('::after')?.style.setProperty('left', `${position}%`);
    
    // CSS 변수로 위치 설정
    barEl.style.cssText = `--pos: ${position}%`;
    
    // ::after 위치 조정을 위한 인라인 스타일
    const indicator = document.createElement('div');
    indicator.className = 'bar-indicator';
    indicator.style.left = `${position}%`;
    barEl.innerHTML = '';
    barEl.appendChild(indicator);
}

// ============================================
// 바 위치 계산
// ============================================
function calculateBarPosition(metricKey, value) {
    const ranges = {
        'elbow': { min: 120, max: 190, goodMin: 155, goodMax: 175 },
        'height': { min: 0, max: 1, goodMin: 0.3, goodMax: 0.7 },
        'hip': { min: 0, max: 0.6, goodMin: 0.15, goodMax: 0.4 },
        'followthrough': { min: 0, max: 0.8, goodMin: 0.25, goodMax: 0.6 }
    };
    
    const range = ranges[metricKey];
    if (!range) return 50;
    
    // 0~100% 범위로 정규화
    const normalized = (value - range.min) / (range.max - range.min);
    return Math.max(0, Math.min(100, normalized * 100));
}

// ============================================
// 구간 점수 계산
// ============================================
function calculatePhaseScore(metrics) {
    const scores = [];
    Object.values(metrics).forEach(m => {
        if (m && m.score) scores.push(m.score);
    });
    
    if (scores.length === 0) return 0;
    return Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
}

// ============================================
// 동시 재생 기능
// ============================================
function syncPlayVideos(phase) {
    const userVideo = document.getElementById(`${phase}-user-video`);
    const expertVideo = document.getElementById(`${phase}-expert-video`);
    
    // 처음으로 이동
    userVideo.currentTime = 0;
    expertVideo.currentTime = 0;
    
    // 동시 재생
    userVideo.play();
    expertVideo.play();
}

// ============================================
// 에러 표시
// ============================================
function showError(message) {
    const content = document.querySelector('.report-content');
    content.innerHTML = `
        <div class="error-message">
            <p>😢 ${message}</p>
            <button onclick="location.href='03-swingMode.html'" 
                    style="margin-top:20px; padding:12px 24px; background:#025B36; color:#fff; border:none; border-radius:10px; cursor:pointer;">
                다시 시도하기
            </button>
        </div>
    `;
}

// ============================================
// 테스트용 더미 데이터
// ============================================
function displayDummyResult() {
    const dummyResult = {
        success: true,
        analysis_id: 'test123',
        overall: {
            score: 72,
            grade: 'good',
            feedback_summary: [
                '[팔꿈치 신전] 팔을 더 쭉 펴주세요! 힘이 제대로 전달되지 않아요.',
                '[골반 회전] 골반 회전 좋아요! 하체 힘이 잘 전달되고 있어요 👍'
            ]
        },
        phases: {
            phase1_ready: {
                name: '준비자세',
                display_type: 'image',
                file_url: null // 테스트용이라 없음
            },
            phase2_backswing_impact: {
                name: '백스윙~임팩트',
                display_type: 'video',
                file_url: null,
                metrics: {
                    elbow_angle: {
                        name: '팔꿈치 신전',
                        value: 145,
                        unit: '°',
                        grade: 'fair',
                        score: 65,
                        feedback: '팔을 더 쭉 펴주세요! 힘이 제대로 전달되지 않아요.'
                    },
                    impact_height: {
                        name: '임팩트 높이',
                        value: 0.52,
                        unit: '',
                        grade: 'good',
                        score: 85,
                        feedback: '타점 높이 좋아요! 👍'
                    },
                    hip_rotation: {
                        name: '골반 회전',
                        value: 28,
                        unit: '%',
                        grade: 'good',
                        score: 82,
                        feedback: '골반 회전 좋아요! 하체 힘이 잘 전달되고 있어요 👍'
                    }
                }
            },
            phase3_followthrough: {
                name: '팔로우스루',
                display_type: 'video',
                file_url: null,
                metrics: {
                    followthrough: {
                        name: '팔로우스루',
                        value: 42,
                        unit: '%',
                        grade: 'good',
                        score: 88,
                        feedback: '팔로우스루 완벽해요! 👍'
                    }
                }
            }
        }
    };
    
    displayResult(dummyResult);
    
    // 테스트 모드 안내
    console.log('📋 테스트 모드: 더미 데이터로 표시 중');
}

// ============================================
// 바 인디케이터 스타일 (동적 생성)
// ============================================
const style = document.createElement('style');
style.textContent = `
    .bar-indicator {
        position: absolute;
        width: 16px;
        height: 16px;
        background: #fff;
        border: 3px solid #025B36;
        border-radius: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        z-index: 10;
    }
    
    .metric-bar {
        position: relative;
    }
`;
document.head.appendChild(style);
