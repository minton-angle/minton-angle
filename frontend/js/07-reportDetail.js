/**
 * 07-reportDetail.js
 * 스윙 분석 리포트 상세 페이지
 */

// ============================================
// 설정
// ============================================
const API_BASE_URL = 'http://localhost:8000';

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
    const postIdx = localStorage.getItem('analysis_post_id');
    
    if (!postIdx) {
        console.log('❌ post_idx 없음');
        showError('분석 결과를 찾을 수 없습니다.');
        return;
    }
    
    console.log('📊 분석 결과 조회:', postIdx);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/upload/result/${postIdx}`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ 결과 로드:', result);
        
        displayResult(result);
        
    } catch (e) {
        console.error('❌ API 호출 오류:', e);
        showError('결과를 불러오는 중 오류가 발생했습니다.');
    }
}

// ============================================
// 결과 표시
// ============================================
function displayResult(result) {
    console.log('🎨 결과 표시:', result);
    
    // success 체크
    if (result.success === false) {
        showError(result.message || '분석에 실패했습니다.');
        return;
    }
    
    // 종합 점수 표시
    displayOverallScore(result.total_score || 0);
    
    // ⭐ files 또는 images/videos 처리
    if (result.files) {
        // 방법 1: files 객체가 있으면
        console.log('📁 files 사용:', result.files);
        displayFiles(result.files);
    } else if (result.images || result.videos) {
        // 방법 2: images/videos 객체면 변환
        console.log('📁 images/videos 변환');
        const files = {
            kf1_image: result.images?.KF1?.path,
            kf2_image: result.images?.KF2?.path,
            kf3_image: result.images?.KF3?.path,
            backswing_video: result.videos?.BACKSWING?.path,
            impact_video: result.videos?.IMPACT?.path
        };
        console.log('📁 변환된 files:', files);
        displayFiles(files);
    } else {
        console.error('❌ files 데이터 없음');
        showError('파일 데이터를 찾을 수 없습니다.');
    }
    
    console.log('✅ 화면 표시 완료');
}

// ============================================
// 종합 점수 표시
// ============================================
function displayOverallScore(score) {
    const scoreEl = document.getElementById('overall-score');
    scoreEl.textContent = score || 0;
    
    const gradeEl = document.getElementById('overall-grade');
    const commentEl = document.getElementById('overall-comment');
    
    // 점수별 등급
    let gradeText, comment;
    
    if (score >= 90) {
        gradeText = '완벽해요! 🎉';
        comment = '전문가 수준의 스윙이에요!';
    } else if (score >= 80) {
        gradeText = '좋아요! 👍';
        comment = '조금만 더 연습하면 완벽해질 거예요.';
    } else if (score >= 70) {
        gradeText = '보통이에요';
        comment = '몇 가지 개선점을 확인해보세요.';
    } else {
        gradeText = '연습이 필요해요';
        comment = '아래 피드백을 참고해서 연습해보세요.';
    }
    
    gradeEl.textContent = gradeText;
    commentEl.textContent = comment;
    
    // SVG 원 애니메이션
    const meter = document.getElementById('score-meter');
    if (meter) {
        const circumference = 2 * Math.PI * 45;
        const offset = circumference - (score / 100) * circumference;
        meter.style.strokeDasharray = `${circumference} ${circumference}`;
        meter.style.strokeDashoffset = offset;
    }
}

// ============================================
// 파일 표시
// ============================================
function displayFiles(files) {
    if (!files) {
        console.log('❌ 파일 없음');
        return;
    }
    
    console.log('📁 파일 표시:', files);
    
    // 1. 준비자세 이미지 (KF1)
    if (files.kf1_image) {
        const img = document.getElementById('phase1-user-img');
        const src = `${API_BASE_URL}${files.kf1_image}`;
        console.log('🖼️ KF1 설정:', src);
        
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        
        img.onload = () => {
            console.log('✅ KF1 이미지 로드 성공');
        };
        img.onerror = (e) => {
            console.error('❌ KF1 이미지 로드 실패:', src, e);
        };
    } else {
        console.log('⚠️ kf1_image 없음');
    }
    
    // 2. 백스윙~임팩트 동영상 (BACKSWING)
    if (files.backswing_video) {
        const video = document.getElementById('phase2-user-video');
        const source = video.querySelector('source');
        const src = `${API_BASE_URL}${files.backswing_video}`;
        console.log('🎥 BACKSWING 설정:', src);
        
        source.src = src;
        video.load();
        
        video.addEventListener('loadedmetadata', () => {
            console.log('✅ BACKSWING 동영상 로드 성공');
        });
        video.addEventListener('error', (e) => {
            console.error('❌ BACKSWING 동영상 로드 실패:', src, e);
        });
    } else {
        console.log('⚠️ backswing_video 없음');
    }
    
    // 3. 임팩트 순간 이미지 (KF3)
    if (files.kf3_image) {
        const img = document.getElementById('phase2-impact-img');
        const src = `${API_BASE_URL}${files.kf3_image}`;
        console.log('🖼️ KF3 설정:', src);
        
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        
        img.onload = () => {
            console.log('✅ KF3 이미지 로드 성공');
        };
        img.onerror = (e) => {
            console.error('❌ KF3 이미지 로드 실패:', src, e);
        };
    } else {
        console.log('⚠️ kf3_image 없음');
    }
    
    // 4. 팔꿈치 최대 신전 이미지 (KF2)
    if (files.kf2_image) {
        const img = document.getElementById('phase2-elbow-img');
        const src = `${API_BASE_URL}${files.kf2_image}`;
        console.log('🖼️ KF2 설정:', src);
        
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        
        img.onload = () => {
            console.log('✅ KF2 이미지 로드 성공');
        };
        img.onerror = (e) => {
            console.error('❌ KF2 이미지 로드 실패:', src, e);
        };
    } else {
        console.log('⚠️ kf2_image 없음');
    }
    
    // 5. 팔로우스루 동영상 (IMPACT)
    if (files.impact_video) {
        const video = document.getElementById('phase3-user-video');
        const source = video.querySelector('source');
        const src = `${API_BASE_URL}${files.impact_video}`;
        console.log('🎥 IMPACT 설정:', src);
        
        source.src = src;
        video.load();
        
        video.addEventListener('loadedmetadata', () => {
            console.log('✅ IMPACT 동영상 로드 성공');
        });
        video.addEventListener('error', (e) => {
            console.error('❌ IMPACT 동영상 로드 실패:', src, e);
        });
    } else {
        console.log('⚠️ impact_video 없음');
    }
}

// ============================================
// 동시 재생 기능 (전문가 영상 준비되면 사용)
// ============================================
function syncPlayVideos(phase) {
    const userVideo = document.getElementById(`${phase}-user-video`);
    const expertVideo = document.getElementById(`${phase}-expert-video`);
    
    if (!userVideo || !expertVideo) {
        console.log('동영상 요소를 찾을 수 없음');
        return;
    }
    
    // 처음으로 이동
    userVideo.currentTime = 0;
    expertVideo.currentTime = 0;
    
    // 동시 재생
    userVideo.play();
    expertVideo.play();
    
    console.log('▶️ 동시 재생:', phase);
}

// ============================================
// 에러 표시
// ============================================
function showError(message) {
    const content = document.querySelector('.report-content');
    if (!content) return;
    
    content.innerHTML = `
        <div style="padding: 40px; text-align: center;">
            <p style="font-size: 18px; color: #666; margin-bottom: 20px;">
                😢 ${message}
            </p>
            <button onclick="location.href='04_1-swingUpload.html'" 
                    style="padding: 12px 24px; background: #025B36; color: #fff; 
                           border: none; border-radius: 10px; cursor: pointer; font-size: 16px;">
                다시 시도하기
            </button>
        </div>
    `;
}

console.log('📄 reportDetail.js 로드 완료');