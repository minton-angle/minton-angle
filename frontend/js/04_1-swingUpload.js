document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('video-input');
    videoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const videoURL = URL.createObjectURL(file);
            const previewVideo = document.getElementById('preview-video');
            const placeholder = document.getElementById('upload-placeholder');
            const previewContainer = document.getElementById('video-preview');
            const submitBtn = document.getElementById('submit-btn');

            previewVideo.src = videoURL;
            placeholder.style.display = 'none';
            previewContainer.style.display = 'block';
            submitBtn.disabled = false; // 버튼 활성화
        }
    });
});

function startAnalysis() {
    const submitBtn = document.getElementById('submit-btn');
    
    // 로딩 상태 표시
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;

    // 1.5초 후 로딩 페이지로 이동 (분석 요청 시뮬레이션)
    setTimeout(() => {
        location.href = '06-reportLoading.html';
    }, 1500);
}