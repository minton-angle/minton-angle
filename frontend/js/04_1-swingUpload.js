let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('video-input');
    videoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            selectedFile = file;
            
            const videoURL = URL.createObjectURL(file);
            const previewVideo = document.getElementById('preview-video');
            const placeholder = document.getElementById('upload-placeholder');
            const previewContainer = document.getElementById('video-preview');
            const submitBtn = document.getElementById('submit-btn');

            previewVideo.src = videoURL;
            previewVideo.play();
            
            placeholder.style.display = 'none';
            previewContainer.style.display = 'block';
            submitBtn.disabled = false;
            
            console.log('📹 선택된 파일:', file.name);
        }
    });
});

// startAnalysis 함수 부분만 교체하시면 됩니다.
async function startAnalysis(event) {
    if (event) {
            event.preventDefault(); 
            event.stopPropagation();
        }
    
    console.log('🚀 분석 시작 버튼 클릭됨! 이제 새로고침 안 됨!');
    
    if (!selectedFile) {
        alert('영상을 먼저 선택해주세요.');
        return;
    }
    
    const submitBtn = document.getElementById('submit-btn');
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    submitBtn.textContent = '서버 분석 중...';
    
    try {
        console.log('🚀 분석 서버로 데이터 전송 시작...');
        
        const formData = new FormData();
        formData.append('video', selectedFile);

        // 타임아웃 넉넉히 5분 (분석이 1분 넘을 때를 대비)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); 
        
        // 🌟 2. authFetch 대신 직접 fetch를 써보세요 (데이터 중복 읽기 에러 방지)
        const response = await fetch(`${API_BASE_URL}/api/upload/video`, {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.detail || `서버 오류 (${response.status})`);
        }
        
        const result = await response.json(); 
        console.log("📥 서버 수신 데이터:", result);

        // 🌟 3. 백엔드 필드명(post_idx)을 최우선으로, 없으면 post_id 사용
        const postId = result.post_idx || result.post_id; 

        if (postId) {
            console.log("🔗 로딩창으로 이동 시도. ID:", postId);
            // 🌟 4. location.assign 보다 window.location.href가 더 안정적일 때가 있습니다.
            window.location.href = `./06-reportLoading.html?post_id=${postId}&type=video`;
        } else {
            console.error("❌ ID 누락됨. 서버 응답 전체:", result);
            alert("서버 응답에서 ID를 찾을 수 없습니다.");
            submitBtn.disabled = false;
            submitBtn.textContent = '분석 시작';
        }
                
    } catch (error) {
        console.error('❌ 에러 발생 상세:', error);
        
        if (error.name === 'AbortError') {
            alert('분석 시간이 너무 오래 걸려 중단되었습니다.');
        } else {
            alert(`에러: ${error.message}`);
        }
        
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
        submitBtn.textContent = '분석 시작';
    }
}

console.log('📄 04_1-swingUpload.js 로드 완료');