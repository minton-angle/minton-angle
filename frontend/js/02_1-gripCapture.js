document.addEventListener('DOMContentLoaded', async () => {
    const webcamContainer = document.getElementById('webcamContainer');
    const shutterBtn = document.getElementById('shutterBtn'); 
    const reCaptureBtn = document.getElementById('reCaptureBtn');
    const resultArea = document.getElementById('resultArea');
    const emptyResult = document.getElementById('emptyResult');
    const myGripBox = document.getElementById('myGrip');
    const loading = document.getElementById('loading');
    const youtubeModal = document.getElementById('youtubeModal');
    const player = document.getElementById('player');

    let video = document.createElement('video');
    let stream = null;

    async function startCamera() {
        const constraints = {
            video: { facingMode: { ideal: "environment" }, width: { ideal: 720 }, height: { ideal: 720 } },
            audio: false
        };
        try {
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            video.autoplay = true;
            video.playsinline = true;
            video.style.width = '100%';
            video.style.height = '100%';
            video.style.objectFit = 'cover';
            webcamContainer.appendChild(video);
        } catch (e) {
            alert("카메라 권한을 확인해주세요.");
        }
    }

    await startCamera();

    shutterBtn.addEventListener('click', async () => {
        // 1. 즉시 캡처 (640 사이즈 리사이징으로 인식률 최적화)
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 640;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 640, 640);
        const capturedImgData = canvas.toDataURL('image/jpeg', 0.9);
        
        // 2. UI 즉시 전환
        loading.style.display = 'flex'; 
        emptyResult.style.display = 'none';

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'grip.jpg');

            try {
                const res = await fetch('https://hoofed-shantell-superaffluently.ngrok-free.dev/api/grip/analyze', { 
                    method: 'POST', 
                    body: formData 
                });
                const data = await res.json();
                handleInference(data, capturedImgData);
            } catch (err) {
                alert("서버 연결 실패");
            } finally {
                loading.style.display = 'none';
            }
        }, 'image/jpeg');
    });

    function handleInference(result, imgData) {
        const feedback = {
            0: { t: "✅ 올바른 그립", d: "훌륭합니다! 정석적인 그립입니다.", color: "#1C593F" },
            1: { t: "❌ 검지 과다 상승", d: "검지가 너무 위로 올라갔어요. 손가락을 내려주세요.", color: "#FF4D4D" },
            2: { t: "❌ 손가락 순서 오류", d: "엄지와 검지의 위치를 확인하세요.", color: "#FF4D4D" },
            3: { t: "❌ 테니스 그립", d: "악수하듯 가볍게 쥐어주세요.", color: "#FF4D4D" },
            4: { t: "❌ 엄지 위치 오류", d: "엄지가 사선을 누르도록 조정하세요.", color: "#FF4D4D" },
            5: { t: "❓ 판독 불가", d: "그립이 명확하지 않습니다. 다시 촬영해주세요.", color: "#888888" }
        };

        const res = feedback[result.class_id] || feedback[5];

        video.style.display = 'none';
        shutterBtn.style.display = 'none';
        document.querySelector('.camera-guide').style.display = 'none';
        reCaptureBtn.style.display = 'block';

        // 바운딩 박스 그리기
        const drawCanvas = document.createElement('canvas');
        const drawCtx = drawCanvas.getContext('2d');
        const img = new Image();

        img.onload = function() {
            drawCanvas.width = 640;
            drawCanvas.height = 640;
            drawCtx.drawImage(img, 0, 0, 640, 640);

            if (result.box && result.box.length === 4 && result.class_id !== 5) {
                const [x1, y1, x2, y2] = result.box;
                drawCtx.strokeStyle = res.color;
                drawCtx.lineWidth = 6;
                drawCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                drawCtx.fillStyle = res.color;
                drawCtx.fillRect(x1, y1 - 30, 150, 30);
                drawCtx.fillStyle = "white";
                drawCtx.font = "bold 20px Pretendard";
                drawCtx.fillText(res.t, x1 + 5, y1 - 8);
            }

            myGripBox.innerHTML = '';
            const resultImg = document.createElement('img');
            resultImg.src = drawCanvas.toDataURL('image/jpeg');
            resultImg.style.width = '100%';
            myGripBox.appendChild(resultImg);
        };
        img.src = imgData;

        resultArea.style.display = 'block'; 
        document.getElementById('resTitle').innerText = res.t;
        document.getElementById('resDesc').innerText = res.d;

        // 오답일 경우 유튜브 팝업 띄우기
        if (result.class_id !== 0 && result.class_id !== 5) {
            setTimeout(() => {
                const videoId = "ZprxwHNp0c8"; // 팝업에서 재생할 유튜브 영상 ID
                player.innerHTML = `
                    <iframe src="https://www.youtube.com/embed/${videoId}?autoplay=1" 
                            frameborder="0" allow="autoplay; encrypted-media" allowfullscreen>
                    </iframe>`;
                youtubeModal.style.display = 'flex';
            }, 800);
        }
    }

    // 팝업 닫기 로직
    document.getElementById('closeModal').addEventListener('click', () => {
        youtubeModal.style.display = 'none';
        player.innerHTML = ''; // 영상 정지
    });

    reCaptureBtn.addEventListener('click', () => location.reload());
});