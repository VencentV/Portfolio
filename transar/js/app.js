let handBoundingBox = { x: 0, y: 0, width: 0, height: 0 };
let currentGesture = '';
let currentProbabilities = [];
let aslModel;
let handposeModel;
let isMirroring = localStorage.getItem('enableMirroring') === 'true'; // Load mirroring setting

window.onload = async function() {
  const video = document.getElementById('video');
  const output = document.getElementById('output');
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');
  const modelInfo = document.getElementById('model-info');
  const mirrorInfo = document.getElementById('mirror-info');

  if (isMirroring) {
    video.classList.add('flipped');
  }

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

  async function loadHandposeModel() {
    handposeModel = await handpose.load();
  }

  async function loadModel(size) {
    try {
      console.log(`Attempting to load model ${size} from:`, `/model/${size}/tfjs_model/model.json`);
      aslModel = await tf.loadLayersModel(`model/${size}/tfjs_model (${size})/model.json`);
      console.log(`Model ${size} loaded successfully`);
    } catch (error) {
      console.error(`Failed to load ASL model ${size}`, error);
      alert(`Failed to load model ${size}: ` + error.message);
      throw new Error(`Failed to load ASL model ${size}: ` + error.message);
    }
  }

  try {
    await setupCamera();
    await loadHandposeModel();
    const selectedModel = localStorage.getItem('selectedModel') || '500'; // Load default model if none selected
    modelInfo.textContent = `Model: ${selectedModel}`;
    mirrorInfo.textContent = `Mirror: ${isMirroring ? 'On' : 'Off'}`;
    await loadModel(selectedModel);
    detectHands();
  } catch (e) {
    handleError(e);
  }

  function handleError(error) {
    console.error(error);
    output.textContent = 'Failed to load video or model: ' + error;
  }

  async function detectHands() {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear previous drawings
    const predictions = await handposeModel.estimateHands(video);
    if (predictions.length > 0) {
      const hand = predictions.reduce((prev, current) => (prev.landmarks[0][0] < current.landmarks[0][0] ? prev : current));
      drawHand(hand);
      const { gesture, probabilities } = await recognizeASLGesture(hand);
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
    requestAnimationFrame(detectHands);
  }

  function drawHand(hand) {
    const [x1, y1] = hand.boundingBox.topLeft;
    const [x2, y2] = hand.boundingBox.bottomRight;
    const width = x2 - x1;
    const height = y2 - y1;

    if (isMirroring) {
      const mirroredX1 = canvas.width - x1 - width;
      handBoundingBox = { x: mirroredX1, y: y1, width, height };
      ctx.strokeRect(mirroredX1, y1, width, height);
      hand.landmarks.forEach(point => {
        const mirroredX = canvas.width - point[0];
        ctx.fillRect(mirroredX, point[1], 5, 5);
      });
    } else {
      handBoundingBox = { x: x1, y: y1, width, height };
      ctx.strokeRect(x1, y1, width, height);
      hand.landmarks.forEach(point => {
        ctx.fillRect(point[0], point[1], 5, 5);
      });
    }

    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
  }

  async function recognizeASLGesture(hand) {
    const { x, y, width, height } = handBoundingBox;
    captureCanvas.width = 200;
    captureCanvas.height = 200;

    // Draw the captured hand area to the capture canvas
    captureCtx.drawImage(video, x, y, width, height, 0, 0, 200, 200);

    // Get image data from the 200x200 canvas, convert to grayscale and flatten
    const imageData = captureCtx.getImageData(0, 0, 200, 200);
    const pixels = imageData.data;
    const grayscalePixels = [];

    for (let i = 0; i < pixels.length; i += 4) {
        // Convert to grayscale using luminosity formula: 0.21 R + 0.72 G + 0.07 B
        const gray = 0.21 * pixels[i] + 0.72 * pixels[i + 1] + 0.07 * pixels[i + 2];
        grayscalePixels.push(gray);
    }

    // Normalize the pixel values if required by the model training process
    const normalizedPixels = grayscalePixels.map(p => p / 255.0);  // Assuming the model expects pixel values in [0,1]

    const inputTensor = tf.tensor2d([normalizedPixels], [1, 40000]);

    // Get the predictions from the model
    const predictions = await aslModel.predict(inputTensor).data();

    // Check the output shape and values
    console.log(predictions);  // Log the predictions to check the output

    // Find the index of the maximum value (most probable class)
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

    // Log probabilities to ensure correct extraction
    console.log(probabilities);

    return { gesture, probabilities };
}



  function formatProbabilities(probabilities) {
    // Sort probabilities by highest first and take the top 3
    const topProbabilities = probabilities.sort((a, b) => b.probability - a.probability).slice(0, 3);
    return topProbabilities.map(p => `${p.gesture}: ${p.probability}`).join('\n');
  }

  function updateOutput(gesture, probabilities) {
    output.textContent = `Hand detected: ASL Gesture - ${gesture}. Top 3 Probabilities:\n${formatProbabilities(probabilities)}`;
  }
}
