document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.querySelector('.login-form');
    const idInput = loginForm.querySelector('input[type="text"]');
    const pwInput = loginForm.querySelector('input[type="password"]');
    const gotoSignupBtn = document.querySelector('.goto-signup-btn');

    const alphanumericRegex = /^[a-zA-Z0-9]+$/;

    // [핵심] 로그인 처리 통합 함수
    async function performLogin() {
        const idValue = idInput.value.trim();
        const pwValue = pwInput.value.trim();

        if (idValue.length < 4 || !alphanumericRegex.test(idValue)) {
            alert('아이디는 4자 이상의 영문 또는 숫자여야 합니다.');
            return idInput.focus();
        }

        if (pwValue.length < 8 || !alphanumericRegex.test(pwValue)) {
            alert('비밀번호는 8자 이상의 영문 또는 숫자여야 합니다.');
            return pwInput.focus();
        }

        try {
            console.log("DB 데이터 조회 중...");
            const response = await mockLoginDB(idValue, pwValue);

            if (response.success) {
                // ⭐ [가장 중요한 부분] 열쇠를 세션에 저장!
                sessionStorage.setItem('isLoggedIn', 'true');
                sessionStorage.setItem('userName', response.userName);

                alert(`${response.userName}님, 환영합니다!`);
                
                // 페이지 이동 (파일명이 01-home.html 인지 home.html 인지 꼭 확인하세요!)
                window.location.href = '01-home.html'; 
            } else {
                alert(response.message);
            }
        } catch (error) {
            alert('서버와 통신 중 오류가 발생했습니다.');
        }
    }

    // 1. 폼 제출(Submit) 시 실행
    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        performLogin();
    });

    // 2. 엔터키 눌렀을 때 실행
    document.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performLogin();
        }
    });

    // 3. 회원가입 버튼
    if (gotoSignupBtn) {
        gotoSignupBtn.addEventListener('click', () => {
            window.location.href = '11-signUp.html';
        });
    }
});

// Mock DB 함수
function mockLoginDB(id, pw) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const mockUser = { id: "mintun123", pw: "password123", name: "홍길동" };
            if (id === mockUser.id && pw === mockUser.pw) {
                resolve({ success: true, userName: mockUser.name });
            } else {
                resolve({ success: false, message: "아이디 또는 비밀번호가 잘못되었습니다." });
            }
        }, 500);
    });
}