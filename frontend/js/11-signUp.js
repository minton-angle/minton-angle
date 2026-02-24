document.addEventListener('DOMContentLoaded', () => {
    const signupForm = document.querySelector('.signup-form');
    const backBtn = document.querySelector('.back-btn');
    const inputs = signupForm.querySelectorAll('input');
    
    const nameInput = inputs[0];
    const idInput = inputs[1];
    const pwInput = inputs[2];
    const pwConfirmInput = inputs[3];
    
    const genderBtns = document.querySelectorAll('.gender-btn');
    const handBtns = document.querySelectorAll('.hand-btn');
    const checkBtn = document.querySelector('.check-btn');

    let selectedGender = null;
    let selectedHand = null;
    let isIdChecked = false;

    const nameRegex = /^[가-힣a-zA-Z]+$/; 
    const idRegex = /^[a-zA-Z0-9]{4,20}$/; 
    const pwRegex = /^[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]{8,20}$/;

    // 뒤로가기
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            signupForm.reset(); 
            genderBtns.forEach(btn => btn.classList.remove('active'));
            handBtns.forEach(btn => btn.classList.remove('active'));
            window.location.href = '12-login.html';
        });
    }

    // 성별 선택
    genderBtns.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            genderBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedGender = index === 0 ? 'female' : 'male';
        });
    });

    // 손 선택
    handBtns.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            handBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedHand = index === 0 ? 'left' : 'right';
        });
    });

    // ⭐ 아이디 중복 확인 (apiCall 사용)
    checkBtn.addEventListener('click', async () => {
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

        try {
            // ⭐ apiCall 사용!
            const data = await apiCall(`/api/auth/check-id?id=${idInput.value}`, {
                method: 'GET',
                auth: false  // 토큰 없이 호출
            });

            if (data.available) {
                alert('사용 가능한 아이디입니다.');
                isIdChecked = true;
            } else {
                alert(data.message);
                isIdChecked = false;
            }
        } catch (error) {
            alert('서버와 통신 중 오류가 발생했습니다.');
            console.error(error);
        }
    });

    // ⭐ 회원가입 제출 (apiCall 사용)
    signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // 빈칸 검사
        if (nameInput.value.trim() === '') {
            alert('이름을 입력해주세요.');
            return nameInput.focus();
        }
        if (!selectedGender) {
            alert('성별을 선택해주세요.');
            return;
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

        // 형식 검사
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

        try {
            // ⭐ apiCall 사용!
            const data = await apiCall('/api/auth/signup', {
                method: 'POST',
                auth: false,  // 토큰 없이 호출
                body: JSON.stringify({
                    id: idInput.value,
                    password: pwInput.value,
                    name: nameInput.value,
                    sex: selectedGender,
                    hand: selectedHand
                })
            });

            alert(data.message);
            window.location.href = '12-login.html';
        } catch (error) {
            alert(error.message || '회원가입에 실패했습니다.');
            console.error(error);
        }
    });
});