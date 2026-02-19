document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.querySelector('.login-form');
    const idInput = loginForm.querySelector('input[type="text"]');
    const pwInput = loginForm.querySelector('input[type="password"]');
    const gotoSignupBtn = document.querySelector('.goto-signup-btn');

    // 규칙: 영문과 숫자만 허용하는 정규식
    const alphanumericRegex = /^[a-zA-Z0-9]+$/;

    // [이벤트] 로그인 제출
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const idValue = idInput.value.trim();
        const pwValue = pwInput.value.trim();

        // 1. 형식 검사 (글자수 및 영문/숫자 조합)
        if (idValue.length < 4 || !alphanumericRegex.test(idValue)) {
            alert('아이디는 4자 이상의 영문 또는 숫자여야 합니다.');
            return idInput.focus();
        }

        if (pwValue.length < 8 || !alphanumericRegex.test(pwValue)) {
            alert('비밀번호는 8자 이상의 영문 또는 숫자여야 합니다.');
            return pwInput.focus();
        }

        // 2. DB 확인 요청 시뮬레이션
        // 실제로는 fetch('https://api.yourdb.com/login', { ... }) 형태가 됩니다.
        try {
            console.log("DB에 데이터 조회 중...");
            const response = await mockLoginDB(idValue, pwValue);

            if (response.success) {
                alert(`${response.userName}님, 환영합니다!`);
                window.location.href = '01-home.html'; // 로그인 성공 시 이동
            } else {
                // DB에 정보가 없거나 일치하지 않을 때
                alert(response.message);
            }
        } catch (error) {
            alert('서버와 통신 중 오류가 발생했습니다.');
        }
    });

    // [이벤트] 회원가입 버튼 클릭
    if (gotoSignupBtn) {
        gotoSignupBtn.addEventListener('click', () => {
            window.location.href = '11-signUp.html';
        });
    }
});

/**
 * DB 서버 역할을 대신하는 가짜 함수 (Mock API)
 * 나중에 실제 백엔드 URL로 교체하게 됩니다.
 */
function mockLoginDB(id, pw) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // 테스트용 가짜 데이터 (나중에 DB 데이터와 비교하게 됨)
            const mockUser = { id: "mintun123", pw: "password123", name: "홍길동" };

            if (id === mockUser.id && pw === mockUser.pw) {
                resolve({ success: true, userName: mockUser.name });
            } else {
                resolve({ 
                    success: false, 
                    message: "아이디 또는 비밀번호가 잘못되었습니다. (DB에 없는 정보)" 
                });
            }
        }, 800); // 0.8초간 서버 응답을 기다리는 척 함
    });
}