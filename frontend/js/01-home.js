// 페이지 로드 완료 후 실행
document.addEventListener('DOMContentLoaded', () => {
    
    // --- [추가] 로그인한 사용자 이름 가져오기 ---
    // 1. 세션 저장소에서 'userName' 가져오기
    const storedName = sessionStorage.getItem('userName');
    
    // 2. 이름을 표시할 HTML 요소 선택
    const userNameElement = document.querySelector('.user-name');

    // 3. 저장된 이름이 있다면 텍스트 변경 (없으면 기본값 '김고수' 유지)
    if (storedName && userNameElement) {
        userNameElement.textContent = `${storedName} 님`;
    }

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
});