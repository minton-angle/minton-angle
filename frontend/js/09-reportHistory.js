document.addEventListener('DOMContentLoaded', function() {
    // 1. 실시간 시계 업데이트
    function updateClock() {
        const timeEl = document.getElementById('current-time');
        if (!timeEl) return;
        const now = new Date();
        timeEl.innerText = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
    }
    setInterval(updateClock, 1000);
    updateClock();

    // 2. 캘린더 초기화
    var calendarEl = document.getElementById('calendar');
    if (calendarEl) {
        var calendar = new FullCalendar.Calendar(calendarEl, {
            initialView: 'dayGridMonth',
            locale: 'en',
            showNonCurrentDates: false,
            fixedWeekCount: false,
            headerToolbar: { left: 'title', center: '', right: 'prev,next' },
            dayCellDidMount: function(info) {
                // 정확한 오늘 날짜 (2026-02-19)
                const now = new Date();
                const todayStr = now.getFullYear() + '-' + 
                               String(now.getMonth() + 1).padStart(2, '0') + '-' + 
                               String(now.getDate()).padStart(2, '0');
                
                const cellDateStr = info.date.getFullYear() + '-' + 
                                  String(info.date.getMonth() + 1).padStart(2, '0') + '-' + 
                                  String(info.date.getDate()).padStart(2, '0');

                // 요청사항: 19일에 파란색 동그라미 활성화
                if (cellDateStr === todayStr) {
                    const numEl = info.el.querySelector('.fc-daygrid-day-number');
                    if (numEl) {
                        numEl.classList.add('active-dot');
                    }
                }
            }
        });
        calendar.render();
    }
});

// 3. 요청사항: 토글 슬라이더 애니메이션 복구
function moveSlider(direction) {
    const slider = document.getElementById('slider');
    const btnCalendar = document.getElementById('btn-calendar');
    const btnTotal = document.getElementById('btn-total');

    if (direction === 'right') {
        slider.style.transform = 'translateX(100%)';
        btnTotal.classList.add('active');
        btnCalendar.classList.remove('active');
        // 애니메이션 후 페이지 이동
        setTimeout(() => { 
            location.href = '08-reportTotal.html'; 
        }, 300);
    } else {
        slider.style.transform = 'translateX(0)';
        btnCalendar.classList.add('active');
        btnTotal.classList.remove('active');
    }
}