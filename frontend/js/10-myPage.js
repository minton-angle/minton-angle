document.addEventListener('DOMContentLoaded', () => {
    // 1. 세션에서 유저 정보 가져와서 이름 바꾸기
    const savedName = sessionStorage.getItem('userName');
    const userNameElement = document.getElementById('userName');

    // 2. 이름이 존재하면 화면에 반영하기
    if (savedName && userNameElement) {
        userNameElement.textContent = savedName; // 이제 "홍길동"이 들어갑니다!
    } else {
        console.log("세션에 이름이 없네요. 로그인 페이지를 확인해 보세요.");
    }

    // 2. 뒤로가기 버튼
    document.getElementById('backMain')?.addEventListener('click', () => {
        history.back();
    });

    // 3. 회원 탈퇴
    document.getElementById('withdrawBtn')?.addEventListener('click', () => {
        if(confirm('정말 탈퇴하시겠습니까? 모든 정보가 사라집니다.')) {
            // 탈퇴 시 세션 비우고 로그인 페이지로 이동하는 로직이 있으면 좋습니다.
            sessionStorage.clear();
            alert('탈퇴 처리가 완료되었습니다.');
            window.location.href = '12-logIn.html';
        }
    });
});