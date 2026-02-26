// ========================================
// 전역 변수
// ========================================
const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const countdownEl = document.getElementById('countdown-number');
const feedbackEl = document.getElementById('feedback-text');
const progressBar = document.getElementById('analysis-progress');

// ⭐ common.js 함수 사용!
let currentSwing = getData('currentSwing', 1);
let swingResults = getData('swingResults', {});
let isRunning = false;
let capturedFrames = [];
let isCapturing = false;
let currentPoseLandmarks = null;

// ========================================
// 1. TTS(음성 출력) 함수
// ========================================
function speak(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ko-KR';
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
}

// ========================================
// 2. MediaPipe Pose 설정
// ========================================
const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
});

pose.setOptions({ 
    modelComplexity: 2,
    smoothLandmarks: true, 
    minDetectionConfidence: 0.3,
    minTrackingConfidence: 0.3
});

pose.onResults((results) => {
    canvasElement.width = results.image.width;
    canvasElement.height = results.image.height;
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    if (results.poseLandmarks) {
        currentPoseLandmarks = results.poseLandmarks;
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
        drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 2, radius: 4});
    } else {
        currentPoseLandmarks = null;
    }
    
    canvasCtx.restore();
});

// ========================================
// 3. 카메라 설정
// ========================================
const camera = new Camera(videoElement, {
    onFrame: async () => { await pose.send({image: videoElement}); },
    width: 1280, 
    height: 720
});
camera.start();

// ========================================
// 4. 페이지 로드 시 초기화
// ========================================
window.addEventListener('DOMContentLoaded', () => {
    console.log(`📂 저장된 스윙 결과: ${Object.keys(swingResults).length}개`);
    
    const completedSwings = currentSwing - 1;
    progressBar.style.width = `${(completedSwings / 3) * 100}%`;
    console.log(`📊 Progress: ${completedSwings}/3 완료`);
    
    Object.keys(swingResults).forEach(swingNum => {
        const result = swingResults[swingNum];
        console.log(`✅ 스윙 ${swingNum}회 결과 복원:`, result);
        updateUIWithResult(parseInt(swingNum), result);
    });
});

// ========================================
// 5. 자동 시작
// ========================================
setTimeout(() => {
    if (!isRunning) {
        isRunning = true;
        const msg = currentSwing === 1 
            ? "자동으로 분석을 시작합니다!" 
            : `스윙 ${currentSwing}회차부터 이어서 시작합니다!`;
        feedbackEl.innerText = msg;
        speak(msg);
        runSwingRoutine();
    }
}, 2000);

// ========================================
// 6. 스윙 분석 루틴
// ========================================
async function runSwingRoutine() {
    if (currentSwing > 3) {
        // ⭐ common.js 함수 사용!
        const postId = getData('post_id');
        
        removeData('currentSwing');
        removeData('swingResults');
        console.log('🧹 Storage 초기화 완료');
        
        const finalMsg = "모든 분석이 완료되었습니다. 결과 리포트로 이동합니다.";
        feedbackEl.innerText = finalMsg;
        speak(finalMsg);
        
        setTimeout(() => {
            window.location.href = `06-reportLoading.html?post_id=${postId}&type=realtime`;
        }, 2500);
        return;
    }

    // 단계 1: 안내
    const startMsg = `스윙 ${currentSwing}회차 준비하세요.`;
    feedbackEl.innerText = startMsg;
    speak(startMsg);
    await new Promise(r => setTimeout(r, 2000));

    // 단계 2: 카운트다운
    for (let i = 3; i > 0; i--) {
        countdownEl.innerText = i;
        speak(i.toString());
        await new Promise(r => setTimeout(r, 1000));
    }

    // 단계 3: 프레임 캡처 시작
    countdownEl.innerText = "";
    speak("지금 휘두르세요!");
    progressBar.style.width = `${(currentSwing / 3) * 100}%`;
    
    capturedFrames = [];
    isCapturing = true;
    
    const captureInterval = setInterval(() => {
        if (isCapturing) {
            captureFrame();
        }
    }, 100);
    
    await new Promise(r => setTimeout(r, 5000));
    
    isCapturing = false;
    clearInterval(captureInterval);

    console.log(`✅ 캡처 완료: ${capturedFrames.length}개 프레임`);

    // 단계 4: 분석 중
    const analyzingMsg = "자세를 분석하고 있습니다. 잠시만 기다려주세요.";
    feedbackEl.innerText = analyzingMsg;
    speak(analyzingMsg);

    // 단계 5: 백엔드로 전송
    try {
        console.log('🚀 백엔드로 전송 중...');
        const analysisResult = await sendFramesToBackend(currentSwing, capturedFrames);
        console.log('✅ 분석 결과 수신:', analysisResult);
        
        // ⭐ common.js 함수 사용!
        swingResults[currentSwing] = analysisResult;
        saveData('swingResults', swingResults);
        console.log(`💾 스윙 ${currentSwing}회 결과 저장 완료`);
        
        updateUIWithResult(currentSwing, analysisResult);
        
    } catch (error) {
        console.error('❌ 분석 실패:', error);
        feedbackEl.innerText = "분석 실패했지만 계속 진행합니다.";
        speak("분석 실패했지만 계속합니다.");
    } finally {
        console.log(`📝 스윙 ${currentSwing} → ${currentSwing + 1}`);
        currentSwing++;
        // ⭐ common.js 함수 사용!
        saveData('currentSwing', currentSwing);
        await new Promise(r => setTimeout(r, 3500));
        runSwingRoutine();
    }
}

// ========================================
// 7. 프레임 캡처
// ========================================
function captureFrame() {
    if (!currentPoseLandmarks) {
        console.warn('⚠️ pose_landmarks 없음, 프레임 건너뜀');
        return;
    }
    
    const selectedIndices = [
        0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    ];
    
    const keypointNames = [
        'nose',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky', 'right_pinky',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ];
    
    const keypoints = {};
    
    selectedIndices.forEach((idx, i) => {
        const landmark = currentPoseLandmarks[idx];
        const name = keypointNames[i];
        
        keypoints[`${name}_x`] = landmark.x;
        keypoints[`${name}_y`] = landmark.y;
        keypoints[`${name}_z`] = landmark.z;
    });
    
    const frameData = canvasElement.toDataURL('image/jpeg', 0.8);
    
    capturedFrames.push({
        frame_id: capturedFrames.length,
        image: frameData,
        keypoints: keypoints
    });
    
    console.log(`✅ 프레임 ${capturedFrames.length} 캡처 완료`);
}

// ========================================
// 8. 백엔드로 전송
// ========================================
async function sendFramesToBackend(swingNum, frames) {
    try {
        console.log(`📤 API 호출: swing_num=${swingNum}, frames=${frames.length}개`);
        
        const keypoints = frames.map(f => f.keypoints);
        console.log(`🔑 Keypoints: ${keypoints.length}개`);
        
        const response = await apiCall('/api/realtime/analyze-swing', {
            method: 'POST',
            headers: {
                "ngrok-skip-browser-warning": "69420" // 👈 이 줄을 추가하세요!
            },
            body: JSON.stringify({
                user_id: getData('user_id'),
                swing_num: swingNum,
                post_id: swingNum > 1 ? getData('post_id') : null,
                keypoints: keypoints,
                frames: frames.map(f => f.image)
            })
        });
        
        console.log('✅ API 응답:', response);
        
        // ⭐ 1회차에서 post_id 저장
        if (swingNum === 1) {
            const postId = response.post_id || response.post_idx;
            if (postId) {
                saveData('post_id', postId);
                console.log(`💾 post_id 저장: ${postId}`);
            } else {
                console.error('❌ 응답에 post_id 없음!', response);
            }
        }
        
        // ⭐ 응답 반환!
        return response;
        
    } catch (error) {
        console.error('❌ API 호출 실패:', error);
        throw error;
    }
}

// ========================================
// 9. UI 업데이트 (수정본)
// ========================================
function updateUIWithResult(swingNum, result) {
    const row = document.getElementById(`row-${swingNum}`);
    
    if (!row) {
        console.warn(`⚠️ row-${swingNum} 요소를 찾을 수 없습니다.`);
        return;
    }
    
    // 기존 활성화 클래스 모두 제거
    row.querySelectorAll('.status-item').forEach(el => 
        el.classList.remove('active', 'bad', 'normal', 'good')
    );
    
    const avgScore = result.total_score || result.overall_average || 0;
    console.log(`📊 스윙 ${swingNum} 점수:`, avgScore); 
    
    // ⭐ 백엔드(SwingService.py)의 get_quick_feedback 기준과 동일하게 맞춤
    let status = 'bad';
    if (avgScore >= 70) {         // 🌟 80 -> 70으로 수정 (잘함)
        status = 'good';
    } else if (avgScore >= 40) {  // 🌟 60 -> 40으로 수정 (보통)
        status = 'normal';
    }
    
    const target = document.getElementById(`res-${swingNum}-${status}`);
    if (target) {
        target.classList.add('active', status);
        console.log(`🎯 UI 활성화 완료: res-${swingNum}-${status}`);
    }
    
    const feedback = result.quick_feedback || result.feedback || "분석 완료";
    feedbackEl.innerText = `스윙 ${swingNum}회: ${feedback} (${avgScore.toFixed(1)}점)`;
    // speak(feedback); // 루틴에서 이미 말하고 있다면 중복 방지를 위해 주석 처리 가능
}

console.log('📄 05-swingAnalyze.js 로드 완료');
