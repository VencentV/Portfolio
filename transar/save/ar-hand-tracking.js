window.onload = async function() {
  const video = document.getElementById('video');
  const output = document.getElementById('output');
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');

  async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Browser API navigator.mediaDevices.getUserMedia not available');
    }
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    video.srcObject = stream;
    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resolve(video);
      };
    });
    video.play();
  }

  async function loadModel() {
    const model = await handpose.load();
    output.textContent = 'Handpose model loaded. Begin hand detection...';
    return model;
  }

  try {
    await setupCamera();
    const model = await loadModel();
    detectHands(model);
  } catch (e) {
    handleError(e);
  }

  function handleError(error) {
    console.error(error);
    output.textContent = 'Failed to load video or model: ' + error;
  }

  async function detectHands(model) {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear previous drawings
    const predictions = await model.estimateHands(video);
    if (predictions.length > 0) {
      predictions.forEach(hand => {
        drawHand(hand);
        const gesture = recognizeASLGesture(hand);
        output.textContent = `Hand detected: ASL Gesture - ${gesture}`;
      });
    } else {
      output.textContent = 'No hand detected';
    }
    requestAnimationFrame(() => detectHands(model));
  }

  function drawHand(hand) {
    // Draw bounding box
    const [x, y, width, height] = hand.boundingBox.topLeft.concat(hand.boundingBox.bottomRight);
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width - x, height - y);

    // Draw landmarks
    hand.landmarks.forEach(point => {
      ctx.fillStyle = "blue";
      ctx.fillRect(point[0], point[1], 5, 5); // Draw a small blue rectangle for each landmark
    });
  }

  function recognizeASLGesture(hand) {
    // Placeholder for gesture recognition logic
    return "A";  // Simulating recognition of ASL sign 'A'
  }
}
