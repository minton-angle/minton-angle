document.addEventListener('DOMContentLoaded', () => {
    // 1. 세션에서 개별 정보 가져오기
    const savedName = sessionStorage.getItem('userName');
    const savedId = sessionStorage.getItem('userId');
    const savedPw = sessionStorage.getItem('userPw');
    const savedGender = sessionStorage.getItem('userGender');
    const savedHand = sessionStorage.getItem('userHand');

    // 각 input 요소 가져오기
    const nicknameInput = document.getElementById('editNickname');
    const idInput = document.getElementById('editId');
    const pwInput = document.getElementById('editPw');
    const genderInput = document.getElementById('editGender');
    const handInput = document.getElementById('editHand');

    // 2. 세션에 데이터가 있다면 input에 채워넣기
    if (savedName) {
        nicknameInput.value = savedName;
        // 아이디, 비밀번호, 성별 등은 로그인 시 세션에 저장되어 있어야 나타납니다.
        idInput.value = savedId || "mintun123"; // 데이터 없으면 더미값
        pwInput.value = savedPw || "password123";
        genderInput.value = savedGender || "남성";
        handInput.value = savedHand || "오른손";
    } else {
        // 로그인 정보가 아예 없는 경우 (테스트용 기본값)
        nicknameInput.value = "김고수";
        idInput.value = "kim_gosu";
        genderInput.value = "남성";
        handInput.value = "오른손";
    }

    // 3. 가상 키보드 제어 (입력창 클릭 시 노출)
    const inputs = document.querySelectorAll('input');
    const keyboard = document.getElementById('mockKeyboard');

    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            keyboard.style.display = 'block';
        });
    });

    // 화면의 다른 곳 클릭 시 키보드 숨기기
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.input-wrap') && !e.target.closest('#mockKeyboard')) {
            if (keyboard) keyboard.style.display = 'none';
        }
    }, true);
});