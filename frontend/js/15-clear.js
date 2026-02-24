let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const videoInput = document.getElementById('videoInput');
    const loading = document.getElementById('loading');

    uploadArea.addEventListener('click', () => {
        if (loading.style.display === 'flex') return;
        videoInput.click();
    });

    videoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        selectedFile = file;
        startSmashAnalysis();
    });
});

async function startSmashAnalysis() {
    const loading = document.getElementById('loading');
    const emptyResult = document.getElementById('emptyResult');
    const resultArea = document.getElementById('resultArea');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const imagePreview = document.getElementById('imagePreview');
    const previewVideo = document.getElementById('previewVideo');
    const resTitle = document.getElementById('resTitle');
    const resDesc = document.getElementById('resDesc');

    // [Step 1] 분석 시작 시 초기화
    loading.style.display = 'flex';
    emptyResult.style.display = 'none';
    resultArea.style.display = 'none';
    imagePreview.style.display = 'none'; // 분석 중에는 이미지가 보이지 않게 숨김
    uploadPlaceholder.style.display = 'none'; // 아이콘도 숨김

    try {
        await new Promise(resolve => setTimeout(resolve, 2500));
        
        const result = {
            score: 85,
            feedback: "정석 궤적과 85% 일치합니다. 타점이 약간 낮으니, 스윙 시 팔을 조금 더 높게 뻗어 임팩트 타이밍을 앞으로 당겨보세요.",
            analyzed_video_url: URL.createObjectURL(selectedFile) 
        };

        // [Step 2] 분석 완료 후 화면 표시
        
        // 상단: 전문가 이미지 노출
        imagePreview.style.display = 'block'; 

        // 하단 카드: 업로드한 본인 영상 재생
        if (result.analyzed_video_url) {
            previewVideo.src = result.analyzed_video_url;
            previewVideo.play();
        }

        resTitle.textContent = "궤적 일치율: " + result.score + "%";
        resDesc.textContent = result.feedback;

        resultArea.style.display = 'block';
        resultArea.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        console.error(error);
        alert("분석 중 오류가 발생했습니다.");
        emptyResult.style.display = 'block';
        uploadPlaceholder.style.display = 'flex'; // 에러 시 다시 업로드 가능하게 복구
    } finally {
        loading.style.display = 'none';
    }
}

// 모달 전역 함수
function openModal() {
    const modal = document.getElementById('videoModal');
    const video = document.getElementById('modalVideo');
    if (modal && video) {
        modal.style.display = 'flex';
        video.play();
    }
}

function closeModal() {
    const modal = document.getElementById('videoModal');
    const video = document.getElementById('modalVideo');
    if (modal) modal.style.display = 'none';
    if (video) {
        video.pause();
        video.currentTime = 0;
    }
}

window.onclick = function(event) {
    const modal = document.getElementById('videoModal');
    if (event.target == modal) {
        closeModal();
    }
};