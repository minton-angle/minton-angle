// ========================================
// 홈 페이지
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    // ⭐ common.js 함수 사용!
    const userName = getUserName();
    
    // 사용자 이름 표시
    const userNameElement = document.querySelector('.user-name');
    if (userName && userNameElement) {
        userNameElement.textContent = `${userName} 님`;
    }
    
    // 1. 그립 교정 버튼
    const gripBtn = document.querySelector('.btn-grib');
    if (gripBtn) {
        gripBtn.addEventListener('click', () => {
            window.location.href = '02-gripMode.html';
        });
    }
    
    // 2. 기본 스윙 교정 버튼
    const swingBtn = document.querySelector('.btn-pose');
    if (swingBtn) {
        swingBtn.addEventListener('click', () => {
            window.location.href = '03-swingMode.html';
        });
    }
    
    // 네비게이션 바
    const navHome = document.querySelector('.nav-home');
    if (navHome) {
        navHome.addEventListener('click', () => {
            window.location.href = '01-home.html';
        });
    }
    
    const navGym = document.querySelector('.nav-play');
    if (navGym) {
        navGym.addEventListener('click', () => {
            window.location.href = '13-playMode.html';
        });
    }
    
    const navHistory = document.querySelector('.nav-history');
    if (navHistory) {
        navHistory.addEventListener('click', () => {
            window.location.href = '09-reportHistory.html';
        });
    }
    
    const navMyPage = document.querySelector('.nav-myPage');
    if (navMyPage) {
        navMyPage.addEventListener('click', () => {
            window.location.href = '10-myPage.html';
        });
    }
});

console.log('📄 01-home.js 로드 완료');