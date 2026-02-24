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

            // 직접 URL 사용 (common.js의 API_BASE_URL 사용)
            const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                
                // 1. 아이디/비밀번호 틀림 (400, 401, 404 등 클라이언트 에러)
                if (response.status >= 400 && response.status < 500) {
                    // 백엔드에서 보내준 에러 메시지를 깔끔하게 팝업으로 띄움
                    alert(error.detail || '아이디 또는 비밀번호가 잘못되었습니다.');
                    return; // 
                }
                
                // 2. 진짜 서버 에러 (500 등)
                throw new Error('서버 내부 오류가 발생했습니다.');
            }

            // ==========================================
            // 정상 로그인 성공 처리
            // ==========================================
            const data = await response.json();
            
            // common.js 함수 사용!
            saveLoginInfo(data.access_token, data.user_id, data.name);
            
            // user_id 중복 저장은 saveLoginInfo에 이미 있으므로 생략해도 됩니다.
            // localStorage.setItem('user_id', data.user_id); 
            
            alert(data.message || '로그인 되었습니다.');
            window.location.href = '01-home.html';
            
        } catch (error) {
            // 3. 서버가 꺼져있거나, 인터넷이 끊겼을 때 (네트워크 에러)
            alert('서버와 통신할 수 없습니다. 서버 상태를 확인해주세요.');
            console.error('서버 통신 에러:', error);
        }
    });

    if (gotoSignupBtn) {
        gotoSignupBtn.addEventListener('click', () => {
            window.location.href = '11-signUp.html';
        });
    }
});

console.log('📄 12-login.js 로드 완료');