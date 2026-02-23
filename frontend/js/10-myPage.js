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
});