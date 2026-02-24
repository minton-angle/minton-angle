/**
 * 02-gripMode.js
 * 가이드 확인 후 촬영 페이지로 이동하는 역할
 */
document.addEventListener('DOMContentLoaded', () => {
    const nextBtn = document.querySelector('.next-btn');

    if (nextBtn) {
        // 1. 버튼을 항상 활성화 상태(초록색)로 유지
        nextBtn.classList.add('active');
        nextBtn.innerText = "그립 교정하러 가기";

        // 2. 클릭 시 촬영 페이지로 이동
        nextBtn.addEventListener('click', () => {
            location.href = '02_1-gripCapture.html';
        });
    }
});