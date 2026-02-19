const API_BASE_URL = 'http://localhost:8000';

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
    const checkBtn = document.querySelector('.check-btn');

    let selectedGender = null; // 'female' or 'male'
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
    genderBtns.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            genderBtns.forEach(b => {
                b.classList.remove('active');
                b.style.backgroundColor = 'white';
                b.style.color = 'black';
            });
            btn.classList.add('active');
            btn.style.backgroundColor = '#3e5d4f';
            btn.style.color = 'white';
            
            // 0: 여성(female), 1: 남성(male)
            selectedGender = index === 0 ? 'female' : 'male';
        });
    });

    // 3. 아이디 중복 확인
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
            const response = await fetch(`${API_BASE_URL}/api/auth/check-id?id=${idInput.value}`);
            const data = await response.json();

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

    // 4. 회원가입 제출
    signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // --- [A] 빈칸 검사 ---
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

        // --- [B] 형식 검사 ---
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

        // --- [C] API 호출 (회원가입) ---
        try {
            const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    id: idInput.value,
                    password: pwInput.value,
                    name: nameInput.value,
                    sex: selectedGender,
                    hand: null // 추후 추가 가능
                })
            });

            const data = await response.json();

            if (response.ok) {
                alert(data.message);
                window.location.href = '12-logIn.html';
            } else {
                alert(data.detail || '회원가입에 실패했습니다.');
            }
        } catch (error) {
            alert('서버와 통신 중 오류가 발생했습니다.');
            console.error(error);
        }
    });
});