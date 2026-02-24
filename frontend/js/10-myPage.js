document.addEventListener('DOMContentLoaded', () => {
    // 1. 세션에서 로그인 정보 가져오기
    const savedName = sessionStorage.getItem('user_name');
    const savedId = sessionStorage.getItem('user_id');
    const accessToken = sessionStorage.getItem('access_token');

    // [보안] 로그인 정보가 없으면 로그인 페이지로 튕겨내기
    if (!accessToken) {
        alert("로그인이 필요한 서비스입니다.");
        window.location.href = '12-logIn.html';
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

    // 4. 회원 탈퇴
    document.getElementById('withdrawBtn')?.addEventListener('click', async () => {
        if (confirm('정말 탈퇴하시겠습니까?')) {
            try {
                // ⭐ 파이썬 라우터의 @router.delete("/me")와 맞춤
                const response = await apiCall('/api/auth/me', {
                    method: 'DELETE',
                    auth: true 
                });

                alert(response.message);
                sessionStorage.clear();
                window.location.href = '12-logIn.html';
            } catch (error) {
                alert('탈퇴 처리 중 오류가 발생했습니다.');
            }
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