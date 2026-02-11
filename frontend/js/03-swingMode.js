/**
 * 03-swingMode.js
 * 스윙 모드 선택 페이지 및 공통 네비게이션 제어
 */

document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. 스윙 모드 선택 로직 ---
    const realtimeBtn = document.querySelector('.mode-card.realtime') || document.getElementById('realtime-btn');
    const uploadBtn = document.querySelector('.mode-card.upload') || document.getElementById('upload-btn');
    
    // 실시간 레슨 버튼 클릭
    if (realtimeBtn) {
        realtimeBtn.addEventListener('click', () => {
            localStorage.setItem('swing_mode', 'realtime');
            location.href = '04-swingGuide.html';
        });
    }
    
    // 영상 업로드 버튼 클릭
    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => {
            localStorage.setItem('swing_mode', 'upload');
            location.href = '04_1-swingUpload.html';
        });
    }

    // --- 2. [추가] 하단 네비게이션 홈 이동 로직 ---
    // 클래스명 .nav-home을 가진 요소나 첫 번째 nav-item을 클릭하면 홈으로 이동합니다.
    const navHome = document.querySelector('.nav-home');
    const navItemHome = document.querySelector('.nav-item:nth-child(1)'); // 구조상 첫 번째가 홈인 경우

    if (navHome) {
        navHome.addEventListener('click', (e) => {
            e.preventDefault(); // 기본 동작 방지
            location.href = '01-home.html';
        });
    } else if (navItemHome) {
        navItemHome.addEventListener('click', () => {
            location.href = '01-home.html';
        });
    }
});

/**
 * 카드 또는 버튼 클릭 이벤트 (HTML에서 onclick으로 호출하는 경우 대비)
 */
function selectMode(mode) {
    localStorage.setItem('swing_mode', mode);
    
    if (mode === 'realtime') {
        location.href = '04-swingGuide.html';
    } else if (mode === 'upload') {
        location.href = '04_1-swingUpload.html';
    }
}

/**
 * 홈으로 이동 함수 (어디서든 호출 가능)
 */
function goHome() {
    location.href = '01-home.html';
}