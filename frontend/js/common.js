// ========================================
// MINTON-ANGLE 공통 유틸리티
// ========================================

// API 기본 URL
const API_BASE_URL = "http://127.0.0.1:8000";

// ⭐ 개발 모드 설정 (배포 시 false로 변경!)
const DEV_MODE = true;

// ========================================
// 개발 모드 초기화
// ========================================
if (DEV_MODE) {
    console.log('%c🔓 개발 모드 활성화', 'color: orange; font-size: 16px; font-weight: bold;');
    console.log('💡 인증 없이 모든 API 사용 가능');
    console.log('📝 배포 시 DEV_MODE = false로 변경하세요');
}

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
    console.log('✅ 로그인 완료:', userId);
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
    // ⭐ 개발 모드: 로그인 체크 스킵
    if (DEV_MODE) {
        console.log('🔓 [개발 모드] 로그인 체크 스킵');
        return;
    }
    
    const publicPages = [
        '00-onboarding.html',
        '11-signUp.html',
        '12-logIn.html'
    ];
    
    const currentPage = window.location.pathname;
    const isPublicPage = publicPages.some(page => currentPage.includes(page));
    
    if (!isLoggedIn() && !isPublicPage) {
        alert('로그인이 필요한 서비스입니다.');
        window.location.href = '12-logIn.html';
    }
}

/**
 * 로그인 필수 페이지 체크 (개별 페이지에서 사용)
 */
function requireLogin() {
    // ⭐ 개발 모드: 항상 true 반환
    if (DEV_MODE) {
        console.log('🔓 [개발 모드] requireLogin 체크 스킵');
        return true;
    }
    
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
    
    const headers = {
        ...options.headers
    };
    
    // 개발 모드가 아니거나 토큰이 있으면 추가
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    
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
    
    // auth 옵션이 false가 아니면 토큰 추가 시도
    if (options.auth !== false) {
        const token = getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        } else if (DEV_MODE) {
            console.log('🔓 [개발 모드] 토큰 없이 API 호출');
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