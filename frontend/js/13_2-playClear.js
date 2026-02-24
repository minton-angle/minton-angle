document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 스매시 예시 페이지 로드 완료');
    
    const previewVideo = document.getElementById('preview-video');
    const goNextBtn = document.getElementById('go-next-btn'); // 버튼 가져오기

    // 1. 영상 재생 로직
    if (previewVideo) {
        previewVideo.play().catch(error => {
            console.log("자동 재생이 차단되었습니다.");
        });
    }

    // 2. 버튼 클릭 이벤트 연결 (핵심!)
    if (goNextBtn) {
        goNextBtn.addEventListener('click', () => {
            console.log('버튼 클릭됨! 이동을 시작합니다.');
            location.href = '15-clear.html'; 
        });
    }
});