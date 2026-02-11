/**
 * 04_1-swingUpload.js
 * 기본 스윙 영상 업로드 페이지
 */

// ============================================
// 설정
// ============================================
const API_BASE_URL = 'http://localhost:8000/api';
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

// ============================================
// 상태
// ============================================
let selectedFile = null;

// ============================================
// 초기화
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('video-input');
    
    videoInput.addEventListener('change', handleFileSelect);
});

// ============================================
// 파일 선택 처리
// ============================================
function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) return;
    
    // 파일 타입 검증
    if (!file.type.startsWith('video/')) {
        alert('동영상 파일만 업로드할 수 있어요.');
        return;
    }
    
    // 파일 크기 검증
    if (file.size > MAX_FILE_SIZE) {
        alert('파일 크기는 100MB 이하여야 해요.');
        return;
    }
    
    selectedFile = file;
    showPreview(file);
    enableSubmitButton();
}

// ============================================
// 미리보기 표시
// ============================================
function showPreview(file) {
    const uploadBox = document.getElementById('upload-box');
    const placeholder = document.getElementById('upload-placeholder');
    const previewContainer = document.getElementById('video-preview');
    const previewVideo = document.getElementById('preview-video');
    
    // 파일 URL 생성
    const fileURL = URL.createObjectURL(file);
    
    // 비디오 소스 설정
    previewVideo.src = fileURL;
    previewVideo.load();
    
    // UI 전환
    placeholder.style.display = 'none';
    previewContainer.style.display = 'block';
    uploadBox.classList.add('has-file');
    
    // 파일 정보 표시
    addFileInfo(file);
}

// ============================================
// 파일 정보 표시
// ============================================
function addFileInfo(file) {
    // 기존 파일 정보 제거
    const existingInfo = document.querySelector('.file-info');
    if (existingInfo) existingInfo.remove();
    
    // 파일 크기 포맷
    const sizeInMB = (file.size / (1024 * 1024)).toFixed(1);
    
    // 파일 정보 요소 생성
    const fileInfo = document.createElement('div');
    fileInfo.className = 'file-info';
    fileInfo.innerHTML = `
        <span class="filename">${file.name}</span>
        <span class="filesize">${sizeInMB}MB</span>
    `;
    
    // 미리보기 컨테이너에 추가
    document.getElementById('video-preview').appendChild(fileInfo);
}

// ============================================
// 제출 버튼 활성화
// ============================================
function enableSubmitButton() {
    const submitBtn = document.getElementById('submit-btn');
    submitBtn.disabled = false;
}

// ============================================
// 분석 시작
// ============================================
async function startAnalysis() {
    if (!selectedFile) {
        alert('영상을 먼저 선택해주세요.');
        return;
    }
    
    const submitBtn = document.getElementById('submit-btn');
    
    try {
        // 로딩 상태
        submitBtn.disabled = true;
        submitBtn.classList.add('loading');
        submitBtn.textContent = '';
        
        // FormData 생성
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // 서버로 업로드 & 분석 요청
        const response = await fetch(`${API_BASE_URL}/swing/upload/analyze`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '분석에 실패했습니다.');
        }
        
        const result = await response.json();
        
        if (result.success) {
            // 결과 저장
            localStorage.setItem('analysis_result', JSON.stringify(result));
            localStorage.setItem('analysis_id', result.analysis_id);
            
            // 로딩 페이지로 이동
            location.href = '06-reportLoading.html';
        } else {
            throw new Error(result.message || '분석에 실패했습니다.');
        }
        
    } catch (error) {
        console.error('분석 오류:', error);
        alert(`오류가 발생했습니다: ${error.message}`);
        
        // 버튼 복구
        submitBtn.disabled = false;
        submitBtn.classList.remove('loading');
        submitBtn.textContent = '분석 시작';
    }
}