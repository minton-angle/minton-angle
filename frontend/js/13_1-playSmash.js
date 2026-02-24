document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 스매시 예시 페이지 로드 완료');
    
    const previewVideo = document.getElementById('preview-video');
    
    // 페이지 로드 시 영상이 자동으로 잘 재생되도록 한 번 더 확인
    if (previewVideo) {
        previewVideo.play().catch(error => {
            console.log("자동 재생이 차단되었습니다. 사용자의 조작이 필요할 수 있습니다.");
        });
    }
});

/**
 * [수정] 버튼 클릭 시 실제 업로드 페이지로 이동
 */
function goToUpload() {
    // 이동하고 싶은 다음 페이지 경로를 적어주세요.
    // 예: 실제 영상 업로드를 수행하는 04_1-swingUpload.html 로 이동
    location.href = '14-smash.html'; 
}