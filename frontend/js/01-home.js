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
});