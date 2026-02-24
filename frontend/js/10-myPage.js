document.addEventListener('DOMContentLoaded', () => {
    // 1. 세션에서 로그인 정보 가져오기
    const savedName = localStorage.getItem('user_name');
    const savedId = localStorage.getItem('user_id');
    const accessToken = localStorage.getItem('access_token');

    // [보안] 로그인 정보가 없으면 로그인 페이지로 튕겨내기
    if (!accessToken) {
        alert("로그인이 필요한 서비스입니다.");
        window.location.href = '12-login.html';
        return;
    }

    // 2. 이름 반영하기
    const userNameElement = document.getElementById('userName');
    if (savedName && userNameElement) {
        userNameElement.textContent = savedName;
    } else if (userNameElement) {
        userNameElement.textContent = "사용자";
    }

    // 3. 뒤로가기 버튼 (상단바 버튼 ID 확인: backMain 또는 back-btn)
    // HTML에 있는 클래스나 ID에 맞춰 연결합니다.
    document.querySelector('.back-btn')?.addEventListener('click', () => {
        window.location.href = '01-home.html'; // 보통 홈으로 보냅니다.
    });

    // 4. 회원 탈퇴 (최종 보정 버전)
    document.getElementById('withdrawBtn')?.addEventListener('click', async () => {
        if (!confirm('정말 탈퇴하시겠습니까?')) return;

        try {
            // common.js의 apiCall 대신 직접 fetch를 써서 변수를 통제해봅니다.
            const token = localStorage.getItem('access_token');
            
            const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`, // 직접 Bearer를 붙여줍니다.
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (response.ok) {
                alert(data.message || "그동안 이용해주셔서 감사합니다.");
                localStorage.clear();
                window.location.href = '12-login.html';
            } else {
                // 서버가 주는 에러 메시지 (사용자를 찾을 수 없습니다 등)를 직접 확인
                throw new Error(data.detail || "탈퇴 실패");
            }

        } catch (error) {
            console.error('탈퇴 에러 상세:', error);
            alert("탈퇴 중 오류 발생: " + error.message);
        }
    });
    const avatarBox = document.getElementById('avatarBox');
    const imageInput = document.getElementById('imageInput');

    // 5. 프로필 클릭 시 파일 선택창 열기
    avatarBox?.addEventListener('click', () => {
        imageInput.click();
    });

    // 6. 파일 선택 시 미리보기 적용
    imageInput?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            // 이미지 파일인지 확인 (보안 및 오류 방지)
            if (!file.type.startsWith('image/')) {
                alert('이미지 파일만 선택 가능합니다.');
                return;
            }

            const reader = new FileReader();
            reader.onload = (event) => {
                // 선택한 사진으로 배경 이미지 교체
                avatarBox.style.backgroundImage = `url(${event.target.result})`;
                avatarBox.style.backgroundSize = 'cover';
                
                // 만약 내부의 <img> 태그를 사용하고 싶다면 아래처럼 할 수도 있습니다.
                // const profileImg = document.getElementById('profileImg');
                // profileImg.src = event.target.result;
                // profileImg.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // [추가 기능] 선택과 동시에 서버에 업로드하려면 여기에 업로드 함수를 호출하세요.
            // uploadProfileImage(file);
        }
    });
});