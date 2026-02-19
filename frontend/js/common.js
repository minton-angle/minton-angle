// 모든 페이지에서 공통으로 쓸 로그인 체크 함수
function checkAuth() {
    const isLoggedIn = sessionStorage.getItem('isLoggedIn');
    // 현재 페이지가 로그인이나 회원가입, 온보딩 페이지가 아닐 때만 체크
    const isPublicPage = window.location.pathname.includes('12-logIn.html')  
                         window.location.pathname.includes('11-signUp.html') 
                         window.location.pathname.includes('00-onboarding.html');

    if (!isLoggedIn && !isPublicPage) {
        alert('로그인이 필요한 서비스입니다.');
        window.location.href = '12-logIn.html';
    }
}

// 페이지가 로드되면 바로 실행
document.addEventListener('DOMContentLoaded', checkAuth);
