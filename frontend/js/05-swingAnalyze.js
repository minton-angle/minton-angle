const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const countdownEl = document.getElementById('countdown-number');
const feedbackEl = document.getElementById('feedback-text');
const progressBar = document.getElementById('analysis-progress');

let currentSwing = 1;
let isRunning = false;

// 1. TTS(음성 출력) 함수
function speak(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ko-KR';
    // 음성 출력이 겹치지 않게 이전 음성 중단 후 실행
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
}

// 2. MediaPipe Pose 및 왜곡 방지 렌더링
const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
});

pose.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
pose.onResults((results) => {
    canvasElement.width = results.image.width;
    canvasElement.height = results.image.height;
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
        drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 2, radius: 4});
    }
    canvasCtx.restore();
});

// 3. 카메라 설정
const camera = new Camera(videoElement, {
    onFrame: async () => { await pose.send({image: videoElement}); },
    width: 1280, height: 720
});
camera.start();

// 4. 음성 인식 트리거 ("시작" 감지)
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.lang = 'ko-KR';
    recognition.continuous = true;
    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        if (transcript.includes("시작") && !isRunning) {
            isRunning = true;
            runSwingRoutine();
        }
    };
    recognition.start();
}

// 5. 스윙 분석 루틴: 안내 -> 카운트다운 -> 바로 스윙
async function runSwingRoutine() {
    if (currentSwing > 3) {
        const finalMsg = "모든 분석이 완료되었습니다. 결과 리포트로 이동합니다.";
        feedbackEl.innerText = finalMsg;
        setTimeout(() => {
            window.location.href = '06-reportLoading.html';
        }, 2500);
        return;
    }

    // 단계 1: 안내 멘트
    const startMsg = `스윙 ${currentSwing}회차 준비하세요.`;
    feedbackEl.innerText = startMsg;
    speak(startMsg);
    await new Promise(r => setTimeout(r, 2000));

    // 단계 2: 카운트다운 (4, 3, 2, 1)
    for (let i = 4; i > 0; i--) {
        countdownEl.innerText = i;
        speak(i.toString());
        await new Promise(r => setTimeout(r, 1000));
    }

    // 단계 3: 스윙 실시 (GO! 문구 삭제, 촬영 시작 안내 음성만)
    countdownEl.innerText = ""; // 화면에서 숫자 지우기
    speak("지금 휘두르세요!"); // 음성으로만 신호
    progressBar.style.width = `${(currentSwing / 3) * 100}%`;
    await new Promise(r => setTimeout(r, 3000)); // 3초간 촬영

    // 단계 4: 분석 상태 안내
    const analyzingMsg = "자세를 분석하고 있습니다. 잠시만 기다려주세요.";
    feedbackEl.innerText = analyzingMsg;
    speak(analyzingMsg);
    await new Promise(r => setTimeout(r, 2000));

    // 단계 5: 결과 업데이트 및 음성 피드백
    const results = ['bad', 'normal', 'good'];
    const randomRes = results[Math.floor(Math.random() * results.length)];
    updateUI(currentSwing, randomRes);
    
    currentSwing++;
    await new Promise(r => setTimeout(r, 3500)); // 피드백 듣고 다음 회차 대기
    runSwingRoutine();
}

function updateUI(swingNum, status) {
    const row = document.getElementById(`row-${swingNum}`);
    row.querySelectorAll('.status-item').forEach(el => el.classList.remove('active', 'bad', 'normal', 'good'));
    
    const target = document.getElementById(`res-${swingNum}-${status}`);
    target.classList.add('active', status);
    
    // 상태에 따른 텍스트 및 음성 메시지 설정
    let comment = "";
    if (status === 'bad') {
        comment = `스윙 ${swingNum}회 결과는 나쁨입니다. 팔을 더 높게 뻗어보세요.`;
    } else if (status === 'normal') {
        comment = `스윙 ${swingNum}회 결과는 보통입니다. 조금 더 빠르게 휘두르면 좋을 것 같아요.`;
    } else {
        comment = `스윙 ${swingNum}회 결과는 잘함입니다. 완벽한 자세예요!`;
    }
    
    feedbackEl.innerText = comment;
    speak(comment); // 텍스트를 음성으로 읽어줌
}

