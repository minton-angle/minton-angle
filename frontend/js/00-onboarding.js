// 온보딩 페이지 - 2초 후 로그인 페이지로 이동
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        window.location.href = '12-logIn.html';
    }, 2000);
});

console.log('📄 00-onboarding.js 로드 완료');