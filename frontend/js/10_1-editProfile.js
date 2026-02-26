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

    // 🌟 1. 내 정보 불러오기 (하드코딩된 fetch 대신 apiCall 사용)
    try {
        console.log("📡 내 정보 불러오는 중...");
        
        // common.js에 설정된 API_BASE_URL과 ngrok 대응 로직을 그대로 타도록 apiCall 사용
        const userData = await apiCall('/api/auth/me', {
            method: 'GET',
            auth: true // 자동으로 sessionStorage의 토큰을 헤더에 넣어줍니다.
        });

        console.log("🛠️ 서버에서 받은 데이터:", userData);

        if (userData) {
            nameInput.value = userData.name || userData.user_name || "";
            idInput.value = userData.id || userData.user_id || "";
            
            // 성별/손 위치 매칭 (백엔드 값이 'female'/'male'인 경우 대응)
            if (userData.sex) {
                genderInput.value = (userData.sex === 'female' || userData.sex === '여성') ? '여성' : '남성';
            }
            if (userData.hand) {
                handInput.value = (userData.hand === 'left' || userData.hand === '왼손') ? '왼손' : '오른손';
            }
        }
    } catch (error) {
        console.error("🚨 에러 발생:", error);
        // ngrok 터널링 문제일 수 있으므로 에러 메시지 상세 출력
        alert('사용자 정보를 불러오지 못했습니다. 다시 로그인해 주세요.');
    }

    // 🌟 2. 정보 수정 제출
    editForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!nameInput.value || !nameRegex.test(nameInput.value)) {
            return alert('이름 형식을 확인해주세요. (한글/영문만 가능)');
        }

        // 비밀번호 입력했을 때만 검사
        let updateData = {
            name: nameInput.value
        };

        if (pwInput.value) {
            if (!pwRegex.test(pwInput.value)) {
                return alert('비밀번호는 영문/숫자 조합 8~20자여야 합니다.');
            }
            if (pwInput.value !== pwConfirmInput.value) {
                return alert('비밀번호가 일치하지 않습니다.');
            }
            updateData.password = pwInput.value;
        }

        try {
            await apiCall('/api/auth/me', {
                method: 'PUT',
                auth: true,
                body: JSON.stringify(updateData)
            });
            
            alert('정보가 수정되었습니다!');

            // 🌟 핵심: sessionStorage 대신 localStorage를 업데이트합니다!
            const newName = nameInput.value;
            localStorage.setItem('user_name', newName); 
            
            // 혹시 모르니 다른 키값들도 로컬에 같이 업데이트해줍니다.
            localStorage.setItem('name', newName);
            localStorage.setItem('nickname', newName);

            location.href = '10-myPage.html';
        } catch (error) {
            console.error("🚨 수정 실패:", error);
            alert('수정에 실패했습니다: ' + error.message);
        }
    });
});