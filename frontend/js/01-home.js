document.addEventListener('DOMContentLoaded', () => {
    // 1. 필요한 요소 가져오기
    const openBtn = document.querySelector('.info-box'); // 가이드 여는 버튼
    const modal = document.getElementById('guideModal');
    const closeBtn = document.getElementById('closeGuide');
    const slider = document.getElementById('tutorialSlider');
    
    // [핵심 수정] 이벤트를 걸어줄 고정된 부모 요소(wrapper)를 가져옵니다.
    const wrapper = document.querySelector('.tutorial-wrapper');
    const dots = document.querySelectorAll('.t-dot');

    let startX = 0;
    let currentIndex = 0;
    let isDragging = false;

    // 2. 슬라이더 업데이트 함수
    function updateSlider() {
        slider.style.transition = 'transform 0.3s ease-out';
        slider.style.transform = `translateX(-${currentIndex * 100}%)`;
        
        dots.forEach((dot, idx) => {
            dot.classList.toggle('active', idx === currentIndex);
        });
    }

    // 3. 좌표 계산용 (마우스/터치 통합)
    const getX = (e) => e.type.includes('mouse') ? e.pageX : e.touches[0].clientX;

    // 드래그 시작
    const dragStart = (e) => {
        isDragging = true;
        startX = getX(e);
        slider.style.transition = 'none';
    };

    // 드래그 중
    const dragMove = (e) => {
        if (!isDragging) return;
        // 폰 브라우저 뒤로가기 방지
        if (e.cancelable) e.preventDefault();
        
        const currentX = getX(e);
        const diff = currentX - startX;
        // [수정] slider 대신 크기가 고정된 wrapper의 너비를 기준으로 계산해야 안전합니다.
        const movePx = (currentIndex * -wrapper.offsetWidth) + diff;
        slider.style.transform = `translateX(${movePx}px)`;
    };

    // 드래그 끝
    const dragEnd = (e) => {
        if (!isDragging) return;
        isDragging = false;

        const endX = e.type.includes('mouse') ? e.pageX : e.changedTouches[0].clientX;
        const diff = endX - startX;

        // 50px 이상 밀었을 때만 페이지 전환
        if (diff < -50 && currentIndex < dots.length - 1) {
            currentIndex++;
        } else if (diff > 50 && currentIndex > 0) {
            currentIndex--;
        }

        updateSlider();
    };

    // 4. 이벤트 연결 [수정: slider 대신 wrapper에 연결]
    if (wrapper) {
        // 모바일 터치
        wrapper.addEventListener('touchstart', dragStart, { passive: true });
        wrapper.addEventListener('touchmove', dragMove, { passive: false });
        wrapper.addEventListener('touchend', dragEnd);

        // PC 마우스
        wrapper.addEventListener('mousedown', dragStart);
        window.addEventListener('mousemove', dragMove);
        window.addEventListener('mouseup', dragEnd);
    }

    // [추가] 4-1. 하단 점(Indicator) 클릭 이벤트 연결
    dots.forEach((dot, index) => {
        dot.addEventListener('click', () => {
            currentIndex = index; // 클릭한 점의 인덱스로 변경
            updateSlider();       // 슬라이더 이동
        });
    });

    // 5. 모달 열기/닫기
    openBtn?.addEventListener('click', () => {
        currentIndex = 0; // 항상 첫 장부터
        updateSlider();
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    });

    closeBtn?.addEventListener('click', () => {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    });

    // 6. 메인 화면 모드 선택 버튼 이벤트
    const btnGrip = document.querySelector('.btn-grib');
    const btnPose = document.querySelector('.btn-pose');

    btnGrip?.addEventListener('click', () => {
        console.log('그립 교정 모드로 이동!');
        // 실제 페이지가 있다면 아래 코드의 주석을 풀고 경로를 수정해 주세요!
        window.location.href = '02-gripMode.html'; 
    });

    btnPose?.addEventListener('click', () => {
        console.log('기본 스윙 교정 모드로 이동!');
        window.location.href = '03-swingMode.html';
    });

    // 7. 하단 네비게이션 바 버튼 이벤트
    const navHome = document.querySelector('.nav-home');
    const navPlay = document.querySelector('.nav-play');
    const navHistory = document.querySelector('.nav-history');
    const navMyPage = document.querySelector('.nav-myPage');

    navHome?.addEventListener('click', () => {
        console.log('홈 화면 클릭됨');
        // window.location.href = 'home.html';
    });
    
    navPlay?.addEventListener('click', () => {
        console.log('플레이 화면 클릭됨');
        window.location.href = '13-playMode.html';
    });
    
    navHistory?.addEventListener('click', () => {
        console.log('히스토리 화면 클릭됨');
        window.location.href = '09-reportHistory.html';
    });
    
    navMyPage?.addEventListener('click', () => {
        console.log('마이페이지 클릭됨');
        window.location.href = '10-myPage.html';
    });
});