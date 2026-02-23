document.addEventListener('DOMContentLoaded', async () => {
    const editForm = document.getElementById('editProfileForm');
    const nameInput = document.getElementById('editNickname');
    const idInput = document.getElementById('editId');
    const pwInput = document.getElementById('editPw');
    const pwConfirmInput = document.getElementById('editPwConfirm');
    const genderInput = document.getElementById('editGender');
    const handInput = document.getElementById('editHand');

    const nameRegex = /^[가-힣a-zA-Z]+$/; 
    const pwRegex = /^[a-zA-Z0-9]{8,20}$/; 

    // 1. 내 정보 불러오기
    try {
        const userData = await apiCall('/api/auth/me', { method: 'GET', auth: true });
        if (userData) {
            nameInput.value = userData.name;
            idInput.value = userData.id;
            genderInput.value = userData.sex === 'female' ? '여성' : '남성';
            handInput.value = userData.hand === 'left' ? '왼손' : '오른손';
        }
    } catch (error) {
        alert('정보를 불러오지 못했습니다.');
    }

    // 2. 정보 수정 제출
    editForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!nameRegex.test(nameInput.value)) {
            return alert('이름 형식이 올바르지 않습니다.');
        }

        // 비밀번호 입력했을 때만 검사
        if (pwInput.value) {
            if (!pwRegex.test(pwInput.value)) {
                return alert('비밀번호는 영문/숫자 조합 8자 이상이어야 합니다.');
            }
            if (pwInput.value !== pwConfirmInput.value) {
                return alert('비밀번호가 일치하지 않습니다.');
            }
        }

        try {
            await apiCall('/api/auth/update-profile', {
                method: 'PUT',
                auth: true,
                body: JSON.stringify({
                    name: nameInput.value,
                    password: pwInput.value || null
                })
            });
            alert('정보가 수정되었습니다!');
            sessionStorage.setItem('user_name', nameInput.value); // 세션 이름 업데이트
            location.href = '10-myPage.html';
        } catch (error) {
            alert('수정에 실패했습니다: ' + error.message);
        }
    });
});