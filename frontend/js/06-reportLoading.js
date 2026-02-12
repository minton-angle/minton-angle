/**
 * 06-reportLoading.js
 * 로딩 화면 → 리포트 페이지 이동
 */

document.addEventListener('DOMContentLoaded', () => {
    // URL 파라미터 읽기
    const urlParams = new URLSearchParams(window.location.search);
    const postId = urlParams.get('post_id');
    const type = urlParams.get('type');
    
    console.log('📊 로딩 화면:', { postId, type });
    
    if (!postId || !type) {
        console.error('❌ post_id 또는 type 없음');
        alert('잘못된 접근입니다.');
        location.href = '01-home.html';
        return;
    }
    
    // ⭐ 2초 로딩 후 리포트로 이동 (파라미터 전달!)
    setTimeout(() => {
        const reportUrl = `07-reportDetail.html?post_id=${postId}&type=${type}`;
        console.log('🔗 이동:', reportUrl);
        location.href = reportUrl;
    }, 2000);
});

console.log('📄 06-reportLoading.js 로드 완료');