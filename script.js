let video = document.getElementById("video");
let resultDiv = document.getElementById("result");

let model;
let flowerFeatures = {};

// ✅ FINAL FLOWERS
const flowers = ["rose", "lotus", "tulip"];
const samplesPerFlower = 6;

// Start camera
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => alert("Camera access denied"));

// Load model and dataset
async function loadModel() {
  model = await mobilenet.load();
  await loadDataset();
  resultDiv.innerText = "Model & dataset loaded. Ready.";
}

loadModel();

// Load dataset images and AVERAGE features
async function loadDataset() {
  for (let flower of flowers) {
    let featureList = [];

    for (let i = 1; i <= samplesPerFlower; i++) {
      const img = new Image();
      img.src = `dataset/${flower}/${flower[0]}${i}.jpg`;

      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });

      let tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();

      let features = model.infer(tensor, true);
      featureList.push(features);
    }

    flowerFeatures[flower] = tf.mean(tf.stack(featureList), 0);
  }
}

// Capture image and predict
async function capture() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  canvas.getContext("2d").drawImage(video, 0, 0);

  let imgTensor = tf.browser.fromPixels(canvas)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .div(255.0)
    .expandDims();

  let liveFeatures = model.infer(imgTensor, true);

  let bestFlower = "";
  let lowestDistance = Infinity;

  for (let flower in flowerFeatures) {
    let distance = tf.norm(
      tf.sub(flowerFeatures[flower], liveFeatures)
    ).dataSync()[0];

    if (distance < lowestDistance) {
      lowestDistance = distance;
      bestFlower = flower;
    }
  }

  let confidence = (100 / (1 + lowestDistance)).toFixed(2);

  if (confidence < 40) {
    resultDiv.innerHTML = `
      ⚠️ Unable to confidently identify flower<br>
      🔄 Please try again
    `;
    return;
  }

  resultDiv.innerHTML = `
    🌸 Flower: <b>${bestFlower.toUpperCase()}</b><br>
    🔍 Confidence: ${confidence}%
  `;
}
