document.addEventListener('DOMContentLoaded', () => {
    const signupForm = document.querySelector('.signup-form');
    const backBtn = document.querySelector('.back-btn');
    const inputs = signupForm.querySelectorAll('input');
    
    // input 요소 매칭 (HTML 순서 기준)
    const nameInput = inputs[0];
    const idInput = inputs[1];
    const pwInput = inputs[2];
    const pwConfirmInput = inputs[3];
    
    const genderBtns = document.querySelectorAll('.gender-btn');
    const handBtns = document.querySelectorAll('.hand-btn');
    const checkBtn = document.querySelector('.check-btn');

    let isGenderSelected = false;
    let isHandSelected = false;
    let isIdChecked = false;

    // --- 정규 표현식 규칙 ---
    const nameRegex = /^[가-힣a-zA-Z]+$/; 
    const idRegex = /^[a-zA-Z0-9]{4,12}$/; 
    const pwRegex = /^[a-zA-Z0-9]{8,20}$/; 

    // 1. 뒤로가기 버튼 로직
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            signupForm.reset(); 
            genderBtns.forEach(btn => {
                btn.classList.remove('active');
                btn.style.backgroundColor = 'white';
                btn.style.color = 'black';
            });
            window.location.href = '12-logIn.html';
        });
    }

    // 2. 성별 버튼 선택 로직
    genderBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            genderBtns.forEach(b =>
                b.classList.remove('active'));
                /*b.style.backgroundColor = 'white';
                b.style.color = 'black';*/
            btn.classList.add('active');
            /*btn.style.backgroundColor = '#3e5d4f';
            btn.style.color = 'white';*/
            isGenderSelected = true;
        });
    });

    // 3. 주로 사용하는 손(Hand) 토글 로직 [추가]
    handBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            handBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active'); // CSS에서 z-index와 색상을 처리함
            isHandSelected = true;
        });
    });

    // 4. 아이디 중복 확인 시 유효성 검사
    checkBtn.addEventListener('click', () => {
        if (idInput.value.trim() === '') {
            alert('아이디를 입력해주세요.');
            idInput.focus();
            return;
        }
        if (!idRegex.test(idInput.value)) {
            alert('아이디는 영문 또는 숫자로 4~12자 사이로 입력해주세요.');
            idInput.focus();
            return;
        }
        alert('사용 가능한 아이디입니다.');
        isIdChecked = true;
    });

    // 5. 회원가입 제출 시 검사 (빈칸 -> 형식 -> 일치 여부 순서)
    signupForm.addEventListener('submit', (e) => {
        e.preventDefault();

        // --- [A] 빈칸 검사 (위에서 아래로 순서대로) ---
        
        if (nameInput.value.trim() === '') {
            alert('이름을 입력해주세요.');
            return nameInput.focus();
        }

        if (!isGenderSelected) {
            alert('성별을 선택해주세요.');
            return; // 버튼은 focus()가 안 되므로 안내만 하고 멈춤
        }

        if (idInput.value.trim() === '') {
            alert('아이디를 입력해주세요.');
            return idInput.focus();
        }

        if (pwInput.value.trim() === '') {
            alert('비밀번호를 입력해주세요.');
            return pwInput.focus();
        }

        if (pwConfirmInput.value.trim() === '') {
            alert('비밀번호 확인을 입력해주세요.');
            return pwConfirmInput.focus();
        }

        // --- [B] 형식 및 중복 확인 검사 (빈칸이 모두 채워진 후 실행) ---

        if (!nameRegex.test(nameInput.value)) {
            alert('이름은 한글 또는 영문으로만 입력해주세요.');
            return nameInput.focus();
        }

        if (!isIdChecked) {
            alert('아이디 중복 확인을 진행해주세요.');
            return;
        }

        if (!pwRegex.test(pwInput.value)) {
            alert('비밀번호는 영문 또는 숫자를 조합하여 8자 이상 입력해주세요.');
            return pwInput.focus();
        }

        if (pwInput.value !== pwConfirmInput.value) {
            alert('비밀번호가 일치하지 않습니다.');
            return pwConfirmInput.focus();
        }

        // 모든 통과 시
        alert('회원가입이 완료되었습니다!');
        window.location.href = '12-logIn.html';
    });
});