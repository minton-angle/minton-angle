/**
 * 03-swingMode.js
 * 스윙 모드 선택 페이지
 */

document.addEventListener('DOMContentLoaded', () => {
    // 버튼 요소 찾기
    const realtimeBtn = document.querySelector('.mode-card.realtime') || document.getElementById('realtime-btn');
    const uploadBtn = document.querySelector('.mode-card.upload') || document.getElementById('upload-btn');
    
    // 실시간 레슨 버튼 클릭
    if (realtimeBtn) {
        realtimeBtn.addEventListener('click', () => {
            // 모드 저장 (리포트에서 구분용)
            localStorage.setItem('swing_mode', 'realtime');
            location.href = '04-swingGuide.html';
        });
    }
    
    // 영상 업로드 버튼 클릭
    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => {
            // 모드 저장
            localStorage.setItem('swing_mode', 'upload');
            location.href = '04_1-swingUpload.html';
        });
    }
});

// 카드 클릭 이벤트 (onclick으로 직접 호출하는 경우)
function selectMode(mode) {
    localStorage.setItem('swing_mode', mode);
    
    if (mode === 'realtime') {
        location.href = '04-swingGuide.html';
    } else if (mode === 'upload') {
        location.href = '04_1-swingUpload.html';
    }
}