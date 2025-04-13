let handBoundingBox = { x: 0, y: 0, width: 0, height: 0 };
let currentGesture = '';
let currentProbabilities = [];

window.onload = async function() {
  const video = document.getElementById('video');
  const output = document.getElementById('output');
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');

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

  async function loadASLModel() {
    try {
      console.log('Attempting to load model from:', '/model/tfjs_model/model.json');
      const model = await tf.loadLayersModel('model/1000/tfjs_model (1000)/model.json');
      console.log('Model loaded successfully');
      return model;
    } catch (error) {
      console.error('Failed to load ASL model', error);
      alert('Failed to load model: ' + error.message);
      throw new Error('Failed to load ASL model: ' + error.message);
    }
  }

  try {
    await setupCamera();
    const handposeModel = await handpose.load();
    const aslModel = await loadASLModel();
    detectHands(handposeModel, aslModel);
  } catch (e) {
    handleError(e);
  }

  function handleError(error) {
    console.error(error);
    output.textContent = 'Failed to load video or model: ' + error;
  }

  async function detectHands(handposeModel, aslModel) {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear previous drawings
    const predictions = await handposeModel.estimateHands(video);
    if (predictions.length > 0) {
      const hand = predictions.reduce((prev, current) => (prev.landmarks[0][0] < current.landmarks[0][0] ? prev : current));
      drawHand(hand);
      const { gesture, probabilities } = await recognizeASLGesture(hand, aslModel);
      if (gesture !== currentGesture) {
        currentGesture = gesture;
        currentProbabilities = probabilities;
        console.log(`Hand detected: ASL Gesture - ${gesture}`);
        console.log(`Probabilities:\n${formatProbabilities(probabilities)}`);
        updateOutput(gesture, probabilities);
      }
    } else {
      output.textContent = 'No hand detected';
      if (currentGesture !== '') {
        console.log('No hand detected');
      }
      currentGesture = '';
      currentProbabilities = [];
    }
    requestAnimationFrame(() => detectHands(handposeModel, aslModel));
  }

  function drawHand(hand) {
    const [x1, y1] = hand.boundingBox.topLeft;
    const [x2, y2] = hand.boundingBox.bottomRight;
    const width = x2 - x1;
    const height = y2 - y1;

    // Mirror the coordinates
    const mirroredX1 = canvas.width - x1 - width;
    const mirroredX2 = canvas.width - x2 - width;

    handBoundingBox = { x: mirroredX1, y: y1, width, height };

    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(mirroredX1, y1, width, height);

    hand.landmarks.forEach(point => {
      const mirroredX = canvas.width - point[0];
      ctx.fillStyle = "blue";
      ctx.fillRect(mirroredX, point[1], 5, 5); // Draw a small blue rectangle for each landmark
    });
  }

  async function recognizeASLGesture(hand, aslModel) {
    const { x, y, width, height } = handBoundingBox;
    captureCanvas.width = 200;
    captureCanvas.height = 200;

    // Draw the captured hand area to the capture canvas
    captureCtx.drawImage(video, x, y, width, height, 0, 0, 200, 200);

    // Get image data and preprocess it
    const imageData = captureCtx.getImageData(0, 0, 200, 200);
    const imageArray = Array.from(imageData.data);
    const grayImage = imageArray.filter((_, index) => index % 4 === 0); // Assuming grayscale (R values only)
    const inputTensor = tf.tensor2d([grayImage], [1, 200 * 200]);

    const predictions = await aslModel.predict(inputTensor).data();
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const gestureMap = {
      0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
      9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
      17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
      25: 'Z', 26: 'SPACE', 27: 'DELETE', 28: 'NOTHING'
    };

    const gesture = gestureMap[maxIndex] || 'NOTHING';
    const probabilities = Array.from(predictions).map((p, i) => ({
      gesture: gestureMap[i] || 'NOTHING',
      probability: p.toFixed(4)
    }));

    return { gesture, probabilities };
  }

  function formatProbabilities(probabilities) {
    // Sort probabilities by highest first and take the top 3
    const topProbabilities = probabilities.sort((a, b) => b.probability - a.probability).slice(0, 3);
    return topProbabilities.map(p => `${p.gesture}: ${p.probability}`).join('\n');
  }

  function updateOutput(gesture, probabilities) {
    output.textContent = `Hand detected: ASL Gesture - ${gesture}\nTop 3 Probabilities:\n${formatProbabilities(probabilities)}`;
  }
}
