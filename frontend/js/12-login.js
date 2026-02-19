const API_BASE_URL = 'http://localhost:8000';

document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.querySelector('.login-form');
    const idInput = loginForm.querySelector('input[type="text"]');
    const pwInput = loginForm.querySelector('input[type="password"]');
    const gotoSignupBtn = document.querySelector('.goto-signup-btn');

    // 규칙: 영문과 숫자만 허용
    const alphanumericRegex = /^[a-zA-Z0-9]+$/;

    // 로그인 제출
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const idValue = idInput.value.trim();
        const pwValue = pwInput.value.trim();

        // 1. 형식 검사
        if (idValue.length < 4 || !alphanumericRegex.test(idValue)) {
            alert('아이디는 4자 이상의 영문 또는 숫자여야 합니다.');
            return idInput.focus();
        }

        if (pwValue.length < 8 || !alphanumericRegex.test(pwValue)) {
            alert('비밀번호는 8자 이상의 영문 또는 숫자여야 합니다.');
            return pwInput.focus();
        }

        // 2. API 호출 (로그인)
        try {
            const formData = new FormData();
            formData.append('username', idValue); // OAuth2 form-data
            formData.append('password', pwValue);

            const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                // 토큰 저장
                localStorage.setItem('access_token', data.access_token);
                localStorage.setItem('user_id', data.user_id);
                localStorage.setItem('user_name', data.name);

                alert(data.message);
                window.location.href = '01-home.html';
            } else {
                alert(data.detail || '로그인에 실패했습니다.');
            }
        } catch (error) {
            alert('서버와 통신 중 오류가 발생했습니다.');
            console.error(error);
        }
    });

    // 회원가입 버튼
    if (gotoSignupBtn) {
        gotoSignupBtn.addEventListener('click', () => {
            window.location.href = '11-signUp.html';
        });
    }
});