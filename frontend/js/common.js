// ========================================
// MINTON-ANGLE 공통 유틸리티
// ========================================

// API 기본 URL
const API_BASE_URL = 'http://172.31.98.95:8000';

// ========================================
// 인증 관련
// ========================================

/**
 * 토큰 가져오기
 */
function getToken() {
    return sessionStorage.getItem('access_token');
}

/**
 * 사용자 ID 가져오기
 */
function getUserId() {
    return sessionStorage.getItem('user_id');
}

/**
 * 사용자 이름 가져오기
 */
function getUserName() {
    return sessionStorage.getItem('user_name');
}

/**
 * 로그인 상태 확인
 */
function isLoggedIn() {
    return !!getToken();
}

/**
 * 로그인 정보 저장
 */
function saveLoginInfo(accessToken, userId, userName) {
    sessionStorage.setItem('access_token', accessToken);
    sessionStorage.setItem('user_id', userId);
    sessionStorage.setItem('user_name', userName);
}

/**
 * 로그아웃
 */
function logout() {
    sessionStorage.clear();
    alert('로그아웃 되었습니다.');
    window.location.href = '12-logIn.html';
}

/**
 * 모든 페이지에서 공통으로 쓸 로그인 체크 함수
 */
function checkAuth() {
    // 공개 페이지 목록 (로그인 없이 접근 가능)
    const publicPages = [
        '00-onboarding.html',
        '01-home.html',
        '02-gripMode.html',
        '03-swingMode.html',
        '04_1-swingUpload.html',
        '04-swingGuide.html',
        '05-swingAnalyze.html',
        '06-reportLoading.html',
        '07-reportDetaile.html',
        '10_1-editProfile.html',
        '10-myPage.html',
        '11-signUp.html',
        '12-logIn.html',
        
        
    ];
    
    // 현재 페이지가 공개 페이지인지 확인
    const currentPage = window.location.pathname;
    const isPublicPage = publicPages.some(page => currentPage.includes(page));
    
    // 로그인 안 했고, 공개 페이지도 아니면 로그인 페이지로
    if (!isLoggedIn() && !isPublicPage) {
        alert('로그인이 필요한 서비스입니다.');
        window.location.href = '12-logIn.html';
    }
}

/**
 * 로그인 필수 페이지 체크 (개별 페이지에서 사용)
 */
function requireLogin() {
    if (!isLoggedIn()) {
        alert('로그인이 필요합니다.');
        window.location.href = '12-logIn.html';
        return false;
    }
    return true;
}

// ⭐ 페이지 로드 시 자동 체크
document.addEventListener('DOMContentLoaded', checkAuth);

// ========================================
// API 호출 헬퍼
// ========================================

/**
 * 인증 헤더 포함 fetch
 */
async function authFetch(url, options = {}) {
    const token = getToken();
    
    if (!token) {
        throw new Error('로그인이 필요합니다.');
    }
    
    const headers = {
        ...options.headers,
        'Authorization': `Bearer ${token}`
    };
    
    return fetch(url, { ...options, headers });
}

/**
 * JSON API 호출 헬퍼
 */
async function apiCall(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const defaultHeaders = {
        'Content-Type': 'application/json'
    };
    
    const headers = {
        ...defaultHeaders,
        ...options.headers
    };
    
    // auth 옵션이 false가 아니면 토큰 자동 추가
    if (options.auth !== false) {
        const token = getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
    }
    
    const response = await fetch(url, { ...options, headers });
    
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `HTTP ${response.status}`);
    }
    
    return response.json();
}

// ========================================
// Storage 헬퍼
// ========================================

/**
 * sessionStorage에 저장
 */
function saveData(key, value) {
    sessionStorage.setItem(key, JSON.stringify(value));
}

/**
 * sessionStorage에서 가져오기
 */
function getData(key, defaultValue = null) {
    const data = sessionStorage.getItem(key);
    if (!data) return defaultValue;
    
    try {
        return JSON.parse(data);
    } catch {
        return data;
    }
}

/**
 * sessionStorage에서 삭제
 */
function removeData(key) {
    sessionStorage.removeItem(key);
}

// ========================================
// 유틸리티
// ========================================

/**
 * 날짜 포맷 (YYYY-MM-DD)
 */
function formatDate(date = new Date()) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

/**
 * 로딩 표시
 */
function showLoading(message = '처리 중...') {
    console.log(`⏳ ${message}`);
}

/**
 * 로딩 숨김
 */
function hideLoading() {
    console.log('✅ 완료');
}
