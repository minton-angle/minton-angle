// ========================================
// 스윙 모드 선택
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    // ⭐ 실제 HTML 클래스명으로 수정!
    const realtimeBtn = document.querySelector('.btn-live');
    const uploadBtn = document.querySelector('.btn-upload');
    
    // 실시간 레슨
    if (realtimeBtn) {
        realtimeBtn.addEventListener('click', () => {
            saveData('swing_mode', 'realtime');
            location.href = '05-swingAnalyze.html';
            console.log('✅ 실시간 레슨 클릭');
        });
    } else {
        console.warn('⚠️ .btn-live 버튼을 찾을 수 없습니다');
    }
    
    // 영상 업로드
    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => {
            saveData('swing_mode', 'upload');
            location.href = '04_1-swingUpload.html';
            console.log('✅ 영상 업로드 클릭');
        });
    } else {
        console.warn('⚠️ .btn-upload 버튼을 찾을 수 없습니다');
    }
    
    // 네비게이션 홈
    const navHome = document.querySelector('.nav-home');
    
    if (navHome) {
        navHome.addEventListener('click', (e) => {
            e.preventDefault();
            location.href = '01-home.html';
        });
    }
});

// HTML onclick용 함수
function selectMode(mode) {
    saveData('swing_mode', mode);
    
    if (mode === 'realtime') {
        location.href = '05-swingAnalyze.html';
    } else if (mode === 'upload') {
        location.href = '04_1-swingUpload.html';
    }
}

function goHome() {
    location.href = '01-home.html';
}

console.log('📄 03-swingMode.js 로드 완료');