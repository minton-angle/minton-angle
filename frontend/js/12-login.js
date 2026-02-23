document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.querySelector('.login-form');
    const idInput = loginForm.querySelector('input[type="text"]');
    const pwInput = loginForm.querySelector('input[type="password"]');
    const gotoSignupBtn = document.querySelector('.goto-signup-btn');

    const flexibleRegex = /^[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]+$/;

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const idValue = idInput.value.trim().toLowerCase();
        const pwValue = pwInput.value.trim();

        if (idValue.length < 4 || !flexibleRegex.test(idValue)) {
            alert('아이디는 4자 이상의 영문 또는 숫자, 특수문자여야 합니다.');
            return idInput.focus();
        }

        if (pwValue.length < 8 || !flexibleRegex.test(pwValue)) {
            alert('비밀번호는 8자 이상의 영문 또는 숫자, 특수문자여야 합니다.');
            return pwInput.focus();
        }

        try {
            const formData = new FormData();
            formData.append('username', idValue);
            formData.append('password', pwValue);

            // ⭐ 직접 URL 사용! (common.js의 API_BASE_URL 사용)
            const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || '로그인 실패');
            }

            const data = await response.json();
            
            // ⭐ common.js 함수 사용!
            saveLoginInfo(data.access_token, data.user_id, data.name);

            localStorage.setItem('user_id', data.user_id);
            
            alert(data.message);
            window.location.href = '01-home.html';
            
        } catch (error) {
            alert('서버와 통신 중 오류가 발생했습니다: ' + error.message);
            console.error('로그인 에러:', error);
        }
    });

    if (gotoSignupBtn) {
        gotoSignupBtn.addEventListener('click', () => {
            window.location.href = '11-signUp.html';
        });
    }
});

console.log('📄 12-login.js 로드 완료');