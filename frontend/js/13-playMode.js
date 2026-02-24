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


    // 홈 버튼
    const navHome = document.querySelector('.nav-home');
    if (navHome) {
        navHome.addEventListener('click', () => {
            window.location.href = '01-home.html';
        });
    }

    // 연습/체육관 버튼 (13-play.html)
    const navGym = document.querySelector('.nav-play');
    if (navGym) {
        navGym.addEventListener('click', () => {
            window.location.href = '13-playMode.html'; // 실제 파일명으로 수정하세요!
        });
    }

    // 기록 버튼 (08-history.html)
    const navHistory = document.querySelector('.nav-history');
    if (navHistory) {
        navHistory.addEventListener('click', () => {
            window.location.href = '09-reportHistory.html'; // 실제 파일명으로 수정하세요!
        });
    }

    // 설정 버튼 (10-myPage.html)
    const navMyPage = document.querySelector('.nav-myPage');
    if (navMyPage) {
        navMyPage.addEventListener('click', () => {
            window.location.href = '10-myPage.html'; // 실제 파일명으로 수정하세요!
        });
    }

    /**
    * 카드 또는 버튼 클릭 이벤트 (HTML에서 onclick으로 호출하는 경우 대비)
    */
    function selectMode(mode) {
        localStorage.setItem('swing_mode', mode);
    
        if (mode === 'smash') {
            location.href = '13_1-playSmash.html';
        } else if (mode === 'clear') {
            location.href = '13_2-playClear.html';
        }
    }

    /**
    * 홈으로 이동 함수 (어디서든 호출 가능)
    */
    function goHome() {
        location.href = '01-home.html';
    
    }
});