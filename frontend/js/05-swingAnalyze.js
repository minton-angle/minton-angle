// const videoElement = document.getElementById('input_video');
// const canvasElement = document.getElementById('output_canvas');
// const canvasCtx = canvasElement.getContext('2d');
// const countdownEl = document.getElementById('countdown-number');
// const feedbackEl = document.getElementById('feedback-text');
// const progressBar = document.getElementById('analysis-progress');

// let currentSwing = 1;
// let isRunning = false;
// let capturedFrames = [];
// let isCapturing = false;
// const API_BASE_URL = 'http://localhost:8000';
// const USER_ID = 'user_001';

// // 1. TTS(음성 출력) 함수
// function speak(text) {
//     const utterance = new SpeechSynthesisUtterance(text);
//     utterance.lang = 'ko-KR';
//     window.speechSynthesis.cancel();
//     window.speechSynthesis.speak(utterance);
// }

// // 2. MediaPipe Pose
// const pose = new Pose({
//     locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
// });

// pose.setOptions({ 
//     modelComplexity: 1, 
//     smoothLandmarks: true, 
//     minDetectionConfidence: 0.5, 
//     minTrackingConfidence: 0.5 
// });

// pose.onResults((results) => {
//     canvasElement.width = results.image.width;
//     canvasElement.height = results.image.height;
//     canvasCtx.save();
//     canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
//     canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
//     if (results.poseLandmarks) {
//         drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
//         drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 2, radius: 4});
//     }
//     canvasCtx.restore();
// });

// // 3. 카메라 설정
// // const camera = new Camera(videoElement, {
// //     onFrame: async () => { await pose.send({image: videoElement}); },
// //     width: 1280, 
// //     height: 720
// // });
// // camera.start();

// // // 4. 음성 인식
// // const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
// // if (SpeechRecognition) {
// //     const recognition = new SpeechRecognition();
// //     recognition.lang = 'ko-KR';
// //     recognition.continuous = true;
// //     recognition.onresult = (event) => {
// //         const transcript = event.results[event.results.length - 1][0].transcript.trim();
// //         if (transcript.includes("시작") && !isRunning) {
// //             isRunning = true;
// //             runSwingRoutine();
// //         }
// //     };
// //     recognition.start();
// // }

// // 촬영 자동 시작
// // 3. 카메라 설정
// const camera = new Camera(videoElement, {
//     onFrame: async () => { await pose.send({image: videoElement}); },
//     width: 1280, 
//     height: 720
// });
// camera.start();

// // 🆕 4. 카메라 준비되면 자동 시작
// setTimeout(() => {
//     if (!isRunning) {
//         isRunning = true;
//         feedbackEl.innerText = "자동으로 분석을 시작합니다!";
//         runSwingRoutine();
//     }
// }, 15000);  // 15초 후 자동 시작

// // 5. 스윙 분석 루틴
// async function runSwingRoutine() {
//     if (currentSwing > 3) {
//         const finalMsg = "모든 분석이 완료되었습니다. 결과 리포트로 이동합니다.";
//         feedbackEl.innerText = finalMsg;
//         speak(finalMsg);
//         setTimeout(() => {
//             window.location.href = '06-reportLoading.html';
//         }, 2500);
//         return;
//     }

//     // 단계 1: 안내
//     const startMsg = `스윙 ${currentSwing}회차 준비하세요.`;
//     feedbackEl.innerText = startMsg;
//     speak(startMsg);
//     await new Promise(r => setTimeout(r, 2000));

//     // 단계 2: 카운트다운
//     for (let i = 4; i > 0; i--) {
//         countdownEl.innerText = i;
//         speak(i.toString());
//         await new Promise(r => setTimeout(r, 1000));
//     }

//     // 단계 3: 프레임 캡처 시작
//     countdownEl.innerText = "";
//     speak("지금 휘두르세요!");
//     progressBar.style.width = `${(currentSwing / 3) * 100}%`;
    
//     capturedFrames = [];
//     isCapturing = true;
    
//     const captureInterval = setInterval(() => {
//         if (isCapturing) {
//             captureFrame();
//         }
//     }, 100);  // 30fps -> 100ms, 10fps로 줄임
    
//     await new Promise(r => setTimeout(r, 3000));
    
//     isCapturing = false;
//     clearInterval(captureInterval);

//     console.log(`✅ 캡처 완료: ${capturedFrames.length}개 프레임`);

//     // 단계 4: 분석 중
//     const analyzingMsg = "자세를 분석하고 있습니다. 잠시만 기다려주세요.";
//     feedbackEl.innerText = analyzingMsg;
//     speak(analyzingMsg);

//     // 단계 5: 백엔드로 전송
//     // 단계 5: 백엔드로 전송
//     try {
//         console.log('🚀 백엔드로 전송 중...');
//         const analysisResult = await sendFramesToBackend(currentSwing, capturedFrames);
//         console.log('✅ 분석 결과 수신:', analysisResult);
        
//         updateUIWithResult(currentSwing, analysisResult);
        
//     } catch (error) {
//         console.error('❌ 분석 실패:', error);
//         feedbackEl.innerText = "분석에 실패했습니다. 다시 시도해주세요.";
//         speak("분석에 실패했습니다.");
//     } finally {
//         // 🆕 성공이든 실패든 무조건 다음으로
//         console.log(`📝 현재 스윙: ${currentSwing} → ${currentSwing + 1}`);
//         currentSwing++;
//         await new Promise(r => setTimeout(r, 3500));
//         runSwingRoutine();
//     }
// }

// // 프레임 캡처
// function captureFrame() {
//     const frameData = canvasElement.toDataURL('image/jpeg', 0.8);
//     capturedFrames.push({
//         frame_id: capturedFrames.length,
//         image: frameData
//     });
// }

// // 백엔드로 전송
// async function sendFramesToBackend(swingNum, frames) {
//     console.log(`📡 API 호출: ${API_BASE_URL}/api/realtime/analyze-swing`);
//     console.log(`📊 데이터: swing_num=${swingNum}, frames=${frames.length}개`);
    
//     const response = await fetch(`${API_BASE_URL}/api/realtime/analyze-swing`, {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({
//             user_id: USER_ID,
//             swing_num: swingNum,
//             frames: frames
//         })
//     });

//     if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//     }

//     return await response.json();
// }

// // UI 업데이트
// function updateUIWithResult(swingNum, result) {
//     const row = document.getElementById(`row-${swingNum}`);
//     row.querySelectorAll('.status-item').forEach(el => 
//         el.classList.remove('active', 'bad', 'normal', 'good')
//     );
    
//     const avgScore = result.overall_average;
//     let status = 'bad';
    
//     if (avgScore >= 80) {
//         status = 'good';
//     } else if (avgScore >= 60) {
//         status = 'normal';
//     }
    
//     const target = document.getElementById(`res-${swingNum}-${status}`);
//     target.classList.add('active', status);
    
//     const feedback = result.feedback;
//     feedbackEl.innerText = `스윙 ${swingNum}회: ${feedback} (${avgScore.toFixed(1)}점)`;
//     speak(feedback);
// }




// ========================================
// 전역 변수
// ========================================
const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const countdownEl = document.getElementById('countdown-number');
const feedbackEl = document.getElementById('feedback-text');
const progressBar = document.getElementById('analysis-progress');

let currentSwing = parseInt(localStorage.getItem('currentSwing') || '1');
let swingResults = JSON.parse(localStorage.getItem('swingResults') || '{}');
let isRunning = false;
let capturedFrames = [];
let isCapturing = false;
let currentPoseLandmarks = null;  // ⭐ 추가: 현재 MediaPipe landmarks

const API_BASE_URL = 'http://localhost:8000';
const USER_ID = 'user_001';

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
    modelComplexity: 2,              // ⭐ 1 → 2 (더 정확)
    smoothLandmarks: true, 
    minDetectionConfidence: 0.3,     // ⭐ 0.5 → 0.3 (더 민감)
    minTrackingConfidence: 0.3       // ⭐ 0.5 → 0.3 (더 민감)
});

pose.onResults((results) => {
    canvasElement.width = results.image.width;
    canvasElement.height = results.image.height;
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    if (results.poseLandmarks) {
        // ⭐ 현재 landmarks 저장
        currentPoseLandmarks = results.poseLandmarks;
        
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
        drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 2, radius: 4});
    } else {
        // ⭐ 사람 미감지
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
    
    // Progress bar 복원
    const completedSwings = currentSwing - 1;
    progressBar.style.width = `${(completedSwings / 3) * 100}%`;
    console.log(`📊 Progress: ${completedSwings}/3 완료`);
    
    // 이전 분석 결과 UI 복원
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
        // localStorage 초기화
        const postId = localStorage.getItem('post_id');
        
        localStorage.removeItem('currentSwing');
        localStorage.removeItem('swingResults');
        console.log('🧹 localStorage 초기화 완료');
        
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
    }, 100);  // 10fps
    
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
        
        // 결과 저장
        swingResults[currentSwing] = analysisResult;
        localStorage.setItem('swingResults', JSON.stringify(swingResults));
        console.log(`💾 스윙 ${currentSwing}회 결과 저장 완료`);
        
        updateUIWithResult(currentSwing, analysisResult);
        
    } catch (error) {
        console.error('❌ 분석 실패:', error);
        feedbackEl.innerText = "분석 실패했지만 계속 진행합니다.";
        speak("분석 실패했지만 계속합니다.");
    } finally {
        console.log(`📝 스윙 ${currentSwing} → ${currentSwing + 1}`);
        currentSwing++;
        localStorage.setItem('currentSwing', currentSwing);
        await new Promise(r => setTimeout(r, 3500));
        runSwingRoutine();
    }
}

// ========================================
// 7. 프레임 캡처 (⭐ 실제 keypoints 추출)
// ========================================
function captureFrame() {
    // ⭐ MediaPipe landmarks 없으면 건너뜀
    if (!currentPoseLandmarks) {
        console.warn('⚠️ pose_landmarks 없음, 프레임 건너뜀');
        return;
    }
    
    // ⭐ 19개 keypoint 인덱스
    const selectedIndices = [
        0,      // nose
        11, 12, // left_shoulder, right_shoulder
        13, 14, // left_elbow, right_elbow
        15, 16, // left_wrist, right_wrist
        17, 18, // left_pinky, right_pinky
        23, 24, // left_hip, right_hip
        25, 26, // left_knee, right_knee
        27, 28, // left_ankle, right_ankle
        29, 30, // left_heel, right_heel
        31, 32  // left_foot_index, right_foot_index
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
    
    // ⭐ 실제 keypoints 추출
    const keypoints = {};
    
    selectedIndices.forEach((idx, i) => {
        const landmark = currentPoseLandmarks[idx];
        const name = keypointNames[i];
        
        keypoints[`${name}_x`] = landmark.x;
        keypoints[`${name}_y`] = landmark.y;
        keypoints[`${name}_z`] = landmark.z;
        // keypoints[`${name}_visibility`] = landmark.visibility;
    });
    
    // Base64 이미지
    const frameData = canvasElement.toDataURL('image/jpeg', 0.8);
    
    capturedFrames.push({
        frame_id: capturedFrames.length,
        image: frameData,
        keypoints: keypoints  // ⭐ 실제 keypoints
    });
    
    console.log(`✅ 프레임 ${capturedFrames.length} 캡처 완료 (keypoints 포함)`);
}

// ========================================
// 8. 백엔드로 전송
// ========================================
async function sendFramesToBackend(swingNum, frames) {
    console.log(`📡 API 호출: ${API_BASE_URL}/api/realtime/analyze-swing`);
    console.log(`📊 데이터: swing_num=${swingNum}, frames=${frames.length}개`);
    
    // ⭐ 실제 keypoints 추출
    const keypoints = frames.map(f => f.keypoints);
    
    console.log(`🔑 Keypoints: ${keypoints.length}개`);
    if (keypoints.length > 0) {
        console.log(`📦 첫 번째 keypoint 샘플:`, Object.keys(keypoints[0]));
    }
    
    const response = await fetch(`${API_BASE_URL}/api/realtime/analyze-swing`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            user_id: USER_ID,
            swing_num: swingNum,
            post_id: swingNum > 1 ? localStorage.getItem('post_id') : null,
            keypoints: keypoints,  // ⭐ 실제 keypoints
            frames: frames.map(f => f.image)
        })
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ 서버 응답:', errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    
    // 1회차면 post_id 저장
    if (swingNum === 1) {
        const postId = result.post_id || result.post_idx;
        
        if (postId) {
            localStorage.setItem('post_id', postId);
            console.log(`💾 post_id 저장 성공: ${postId}`);
        } else {
            console.error('❌ 응답에 post_id/post_idx 없음!', result);
        }
    }
    
    return result;
}

// ========================================
// 9. UI 업데이트
// ========================================
function updateUIWithResult(swingNum, result) {
    const row = document.getElementById(`row-${swingNum}`);
    
    if (!row) {
        console.warn(`⚠️ row-${swingNum} 요소를 찾을 수 없습니다.`);
        return;
    }
    
    row.querySelectorAll('.status-item').forEach(el => 
        el.classList.remove('active', 'bad', 'normal', 'good')
    );
    
    const avgScore = result.overall_average || result.total_score || 0;
    let status = 'bad';
    
    if (avgScore >= 80) {
        status = 'good';
    } else if (avgScore >= 60) {
        status = 'normal';
    }
    
    const target = document.getElementById(`res-${swingNum}-${status}`);
    
    if (target) {
        target.classList.add('active', status);
    } else {
        console.warn(`⚠️ res-${swingNum}-${status} 요소를 찾을 수 없습니다.`);
    }
    
    const feedback = result.quick_feedback || result.feedback || "분석 완료";
    feedbackEl.innerText = `스윙 ${swingNum}회: ${feedback} (${avgScore.toFixed(1)}점)`;
    speak(feedback);
}