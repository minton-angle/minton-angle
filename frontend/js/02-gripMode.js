/**
 * 02-gripMode.js
 * 그립 가이드 촬영 및 이미지 프리뷰 제어
 */

document.addEventListener('DOMContentLoaded', () => {

    // --- 1. 요소 가져오기 (안전하게) ---
    const cameraBox = document.getElementById('cameraBox');
    const realInput = document.getElementById('realCameraInput');
    const nextBtn = document.querySelector('.next-btn');

    // --- 2. 카메라/이미지 업로드 로직 ---
    if (cameraBox && realInput) {
        
        // 2-1. 박스 클릭 시 -> 숨겨진 파일 버튼 클릭
        cameraBox.addEventListener('click', () => {
            realInput.click();
        });

        // 2-2. 파일이 선택되었을 때 (촬영 완료 시)
        realInput.addEventListener('change', (event) => {
            const file = event.target.files[0];

            if (file) {
                const reader = new FileReader();

                // 파일 읽기 성공 시
                reader.onload = (e) => {
                    // (1) 박스 내용을 촬영된 이미지로 교체
                    cameraBox.innerHTML = `
                        <img src="${e.target.result}" 
                             alt="촬영된 그립 이미지" 
                             style="width: 100%; height: 100%; object-fit: cover; border-radius: 20px;">
                    `;

                    // (2) 다음 버튼 활성화 (스타일 변경 및 상태 업데이트)
                    if (nextBtn) {
                        nextBtn.classList.add('active'); // CSS에서 색상 변경
                        nextBtn.innerText = "다음 단계로 이동"; // 텍스트 변경 (선택사항)
                    }
                };

                // 이미지 읽기 시작
                reader.readAsDataURL(file);
            }
        });
    }

    // --- 3. 하단 '다음' 버튼 이동 로직 ---
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            // (1) 사진을 찍었는지 확인 (validation)
            // realInput에 파일이 없으면 넘어가지 않음
            if (!realInput.files || realInput.files.length === 0) {
                alert("그립 사진을 먼저 촬영해주세요! 📸");
                return;
            }

            // (2) 사진이 있다면 다음 페이지로 이동
            // (03-swingMode.html 로 이동한다고 가정)
            location.href = '03-swingMode.html';
        });
    }
});