const API_BASE_URL = 'http://localhost:8000';
const USER_ID = 'user_001'; // 임시 사용자 ID

let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('video-input');
    videoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            // 파일 저장
            selectedFile = file;
            
            // 미리보기 표시
            const videoURL = URL.createObjectURL(file);
            const previewVideo = document.getElementById('preview-video');
            const placeholder = document.getElementById('upload-placeholder');
            const previewContainer = document.getElementById('video-preview');
            const submitBtn = document.getElementById('submit-btn');

            previewVideo.src = videoURL;
            previewVideo.play(); // 자동 재생
            
            placeholder.style.display = 'none';
            previewContainer.style.display = 'block';
            submitBtn.disabled = false;
            
            console.log('📹 선택된 파일:', file.name);
        }
    });
});

async function startAnalysis() {
    if (!selectedFile) {
        alert('영상을 먼저 선택해주세요.');
        return;
    }
    
    const submitBtn = document.getElementById('submit-btn');
    
    // 로딩 상태 시작
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    submitBtn.textContent = '분석 중...';
    
    try {
        console.log('🚀 분석 시작...');
        
        // FormData 생성
        const formData = new FormData();
        formData.append('video', selectedFile);
        
        // API 호출
        const response = await fetch(`${API_BASE_URL}/api/upload/video?user_id=${USER_ID}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '분석 실패');
        }
        
        const result = await response.json();
        console.log('✅ 분석 완료:', result);
        
        // ⭐ URL 파라미터로 전달 (localStorage 사용 안 함!)
        const postId = result.post_idx;
        
        // 결과 페이지로 이동
        setTimeout(() => {
            location.href = `06-reportLoading.html?post_id=${postId}&type=video`;
        }, 500);
        
    } catch (error) {
        console.error('❌ 에러 발생:', error);
        alert(`분석 실패: ${error.message}`);
        
        // 버튼 복구
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
        submitBtn.textContent = '분석 시작';
    }
}

console.log('📄 04_1-swingUpload.js 로드 완료');