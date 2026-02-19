// 페이지 로드 완료 후 실행
document.addEventListener('DOMContentLoaded', () => {
    
    // 1. 그립 교정 버튼 클릭 시 이동
    const gripBtn = document.querySelector('.btn-grib');
    if (gripBtn) {
        gripBtn.addEventListener('click', () => {
            window.location.href = '02-gripMode.html';
        });
    }

    // 2. 기본 스윙 교정 버튼 클릭 시 이동
    const swingBtn = document.querySelector('.btn-pose');
    if (swingBtn) {
        swingBtn.addEventListener('click', () => {
            window.location.href = '03-swingMode.html';
        });
    }

    // (참고) 네비게이션 바 버튼들도 필요하다면 아래와 같이 추가 가능합니다.
    const navHistory = document.querySelector('.nav-history');
    if (navHistory) {
        navHistory.addEventListener('click', () => {
            // 기록 페이지가 있다면 해당 경로로 이동
            // window.location.href = '08-history.html';
        });
    }
});