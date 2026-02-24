// ========================================
// 전역 변수
// ========================================
let calendar;
let monthlyData = {}; // 월별 리포트 데이터

// ========================================
// 초기화
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('📅 캘린더 페이지 로드');
    
    // 시계 업데이트
    updateClock();
    setInterval(updateClock, 1000);
    
    // 캘린더 초기화
    initCalendar();
});

// ========================================
// 시계
// ========================================
function updateClock() {
    const timeEl = document.getElementById('current-time');
    if (!timeEl) return;
    const now = new Date();
    timeEl.innerText = now.toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit', 
        hour12: true 
    });
}

// ========================================
// 캘린더 초기화
// ========================================
function initCalendar() {
    const calendarEl = document.getElementById('calendar');
    if (!calendarEl) return;
    
    calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: 'dayGridMonth',
        locale: 'en',
        showNonCurrentDates: false,
        fixedWeekCount: false,
        height: '100%',
        contentHeight: '100%',
        expandRows: true,
        stickyHeaderDates: false,
        handleWindowResize: true,
        headerToolbar: { 
            left: 'title', 
            center: '', 
            right: 'prev,next' 
        },
        
        // 월 변경 시
        datesSet: function(info) {
            const year = info.view.currentStart.getFullYear();
            const month = info.view.currentStart.getMonth() + 1;
            console.log(`📅 월 변경: ${year}-${month}`);
            loadMonthlyData(year, month);
        },
        
        // 날짜 셀 렌더링
        dayCellDidMount: function(info) {
            const dateStr = formatDate(info.date);
            const numEl = info.el.querySelector('.fc-daygrid-day-number');
            
            if (!numEl) return;
            
            // 월별 데이터에서 확인
            if (monthlyData[dateStr]) {
                const data = monthlyData[dateStr];
                
                // 기존 점 제거
                const existingDots = numEl.querySelector('.report-dots');
                if (existingDots) {
                    existingDots.remove();
                }
                
                // 점 표시 추가
                const dotContainer = document.createElement('div');
                dotContainer.className = 'report-dots';
                
                if (data.realtime) {
                    const dot = document.createElement('span');
                    dot.className = 'dot realtime-dot';
                    dot.textContent = '🔴';
                    dotContainer.appendChild(dot);
                }
                
                if (data.video) {
                    const dot = document.createElement('span');
                    dot.className = 'dot video-dot';
                    dot.textContent = '📹';
                    dotContainer.appendChild(dot);
                }
                
                numEl.appendChild(dotContainer);
                numEl.classList.add('has-report');
            }
        },
        
        // 날짜 클릭
        dateClick: function(info) {
            const dateStr = formatDate(info.date);
            console.log(`📅 날짜 클릭: ${dateStr}`);
            
            // 리포트 있는 날짜만 모달 열기
            if (monthlyData[dateStr]) {
                openReportModal(dateStr);
            } else {
                alert('해당 날짜에 리포트가 없습니다.');
            }
        }
    });
    
    calendar.render();
}

// ========================================
// 월별 데이터 로드
// ========================================
async function loadMonthlyData(year, month) {
    try {
        console.log(`🔄 월별 데이터 로드 시작: ${year}-${month}`);
        
        const response = await apiCall(
            `/api/calendar/monthly-summary?year=${year}&month=${month}`
        );
        
        if (response.success) {
            monthlyData = response.summary;
            console.log('✅ 월별 데이터 로드 완료:', monthlyData);
            
            // 캘린더 리렌더링
            calendar.refetchEvents();
            
            // 모든 날짜 셀 다시 렌더링
            const allDayCells = document.querySelectorAll('.fc-daygrid-day');
            allDayCells.forEach(cell => {
                const dateStr = cell.getAttribute('data-date');
                if (dateStr && monthlyData[dateStr]) {
                    const numEl = cell.querySelector('.fc-daygrid-day-number');
                    if (numEl && !numEl.classList.contains('has-report')) {
                        const data = monthlyData[dateStr];
                        
                        const dotContainer = document.createElement('div');
                        dotContainer.className = 'report-dots';
                        
                        if (data.realtime) {
                            const dot = document.createElement('span');
                            dot.className = 'dot realtime-dot';
                            dot.textContent = '🔴';
                            dotContainer.appendChild(dot);
                        }
                        
                        if (data.video) {
                            const dot = document.createElement('span');
                            dot.className = 'dot video-dot';
                            dot.textContent = '📹';
                            dotContainer.appendChild(dot);
                        }
                        
                        numEl.appendChild(dotContainer);
                        numEl.classList.add('has-report');
                    }
                }
            });
        }
    } catch (error) {
        console.error('❌ 월별 데이터 로드 실패:', error);
    }
}

// ========================================
// 리포트 모달 열기
// ========================================
async function openReportModal(dateStr) {
    try {
        console.log(`🔄 리포트 조회: ${dateStr}`);
        
        const response = await apiCall(
            `/api/calendar/reports?date=${dateStr}`
        );
        
        if (response.success) {
            showModal(dateStr, response);
        }
    } catch (error) {
        console.error('❌ 리포트 조회 실패:', error);
        alert('리포트를 불러오는데 실패했습니다.');
    }
}

// ========================================
// 모달 표시
// ========================================
function showModal(dateStr, data) {
    // 기존 모달 제거
    const existingModal = document.querySelector('.report-modal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // 모달 HTML 생성
    const modal = document.createElement('div');
    modal.className = 'report-modal';
    modal.innerHTML = `
        <div class="modal-overlay" onclick="closeModal()"></div>
        <div class="modal-content">
            <div class="modal-header">
                <h3>${formatDateKorean(dateStr)} 스윙 리포트</h3>
                <button class="modal-close" onclick="closeModal()">✕</button>
            </div>
            <div class="modal-body">
                ${generateReportCards(data)}
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // 애니메이션을 위한 딜레이
    setTimeout(() => {
        modal.classList.add('show');
    }, 10);
}

// ========================================
// 리포트 카드 생성
// ========================================
function generateReportCards(data) {
    let html = '';
    
    // 실시간 리포트
    if (data.realtime_report) {
        const report = data.realtime_report;
        html += `
            <div class="report-card realtime-card">
                <div class="card-icon">🔴</div>
                <div class="card-content">
                    <h4>실시간 분석</h4>
                    <p class="card-time">⏰ ${report.time}</p>
                    <p class="card-score">점수: <strong>${report.total_score}점</strong></p>
                </div>
                <button class="card-button" onclick="goToReport('${report.post_idx}', 'realtime')">
                    자세히 보기 →
                </button>
            </div>
        `;
    }
    
    // 동영상 리포트
    if (data.video_report) {
        const report = data.video_report;
        html += `
            <div class="report-card video-card">
                <div class="card-icon">📹</div>
                <div class="card-content">
                    <h4>동영상 업로드</h4>
                    <p class="card-time">⏰ ${report.time}</p>
                    <p class="card-score">점수: <strong>${report.total_score}점</strong></p>
                </div>
                <button class="card-button" onclick="goToReport('${report.post_idx}', 'video')">
                    자세히 보기 →
                </button>
            </div>
        `;
    }
    
    // 둘 다 없으면
    if (!data.realtime_report && !data.video_report) {
        html = '<p class="no-report">해당 날짜에 리포트가 없습니다.</p>';
    }
    
    return html;
}

// ========================================
// 모달 닫기
// ========================================
function closeModal() {
    const modal = document.querySelector('.report-modal');
    if (modal) {
        modal.classList.remove('show');
        setTimeout(() => {
            modal.remove();
        }, 300);
    }
}

// ========================================
// 리포트 상세 페이지로 이동
// ========================================
function goToReport(postIdx, type) {
    console.log(`📄 리포트 이동: ${postIdx} (${type})`);
    location.href = `07-reportDetail.html?post_id=${postIdx}&type=${type}`;
}

// ========================================
// 토글 슬라이더
// ========================================
function moveSlider(direction) {
    const slider = document.getElementById('slider');
    const btnCalendar = document.getElementById('btn-calendar');
    const btnTotal = document.getElementById('btn-total');

    if (direction === 'right') {
        slider.style.transform = 'translateX(100%)';
        btnTotal.classList.add('active');
        btnCalendar.classList.remove('active');
        setTimeout(() => { 
            location.href = '08-reportTotal.html'; 
        }, 300);
    } else {
        slider.style.transform = 'translateX(0)';
        btnCalendar.classList.add('active');
        btnTotal.classList.remove('active');
    }
}

// ========================================
// 유틸리티
// ========================================
function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function formatDateKorean(dateStr) {
    const [year, month, day] = dateStr.split('-');
    return `${year}년 ${parseInt(month)}월 ${parseInt(day)}일`;
}