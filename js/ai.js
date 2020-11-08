let faceapi;
let video;
let width = 320;
let height = 240;
let IMG_WIDTH = 50;
let IMG_HEIGHT = 50;
const videoSrcId = 'srcId';
let isPaused = true;
let feedLoaded = false;
let detectionStarted = false;
let trainingSetStarted = false;
let trainingSetIntervalId = -1;
let currentTrainingSet = [];
let trainingData = [];
let BATCH_SIZE = 50;
let trainingContainerId = 'trainingContainer';
let globalModelRef;
let modelLoaded = false;
let predictionTimer = null;
const modelConfig = {
    withLandmarks: false,
    withDescriptors: false,
    Mobilenetv1Model: 'http://localhost:9090/mlapp/js/models',
    FaceLandmarkModel: 'http://localhost:9090/mlapp/js/models',
    FaceLandmark68TinyNet: 'http://localhost:9090/mlapp/js/models',
    FaceRecognitionModel: 'http://localhost:9090/mlapp/js/models'
  }

  const classNames = ['Happy', 'Sad', 'Neutral'];

window.addEventListener('DOMContentLoaded', function() {
    globalModelRef = makeTFModel();
    init();
  });

class TrainingImage{

    constructor(imgData,width,height,emotion){
        this.imgData = imgData;
        this.width = width;
        this.height = height;
        this.emotion = emotion;
    }

}

async function init(){
    await initVideoStream();
    video =  getVideoStream();
    feedLoaded = true;
    video.width = width;
    video.height = height;
    let hiddenCanvas = document.getElementById('hiddenCanvas');
    let faceCanvas = document.getElementById('faceCanvas');
    hiddenCanvas.width = width;
    hiddenCanvas.height = height;
    faceCanvas.width = IMG_WIDTH;
    faceCanvas.height = IMG_HEIGHT;
    console.log('Video Stream Loaded');
  }

async function toggleDetect(){

    if(!detectionStarted){
    if(feedLoaded && !isPaused){
    detectionStarted = true;
    document.getElementById('toggleDetect').innerText = 'Stop Detect Face';
    faceapi = ml5.faceApi(video, modelConfig, modelReady); 
    }
}
else{
    detectionStarted = false;
    document.getElementById('toggleDetect').innerText = 'Start Detect Face';
}
}

async function toggleCamera(){
    if(feedLoaded){
    if(video.paused){
        isPaused = false;
        document.getElementById('toggleCapture').innerText = 'Stop Capture';
        video.play();
    }
    else{
        isPaused = true;
        document.getElementById('toggleCapture').innerText = 'Start Capture';
        video.pause();
    }
}
else{
    console.log('Video Feed not yet loaded');
}
}


function modelReady() {
    console.log('Face Detect Started')
    faceapi.detect(gotResults)
}


function clearCanvases(hiddenCtx,ctx){
hiddenCtx.clearRect(0,0,hiddenCanvas.width,hiddenCanvas.height);
ctx.clearRect(0,0,faceCanvas.width,faceCanvas.height);
}

// Use getImageData and putImageData to create training set

function drawInCanvas(detection){

    var hiddenCanvas = document.getElementById('hiddenCanvas');
    var faceCanvas = document.getElementById('faceCanvas');
    var hiddenCtx = hiddenCanvas.getContext('2d');
    var ctx = faceCanvas.getContext('2d');
    hiddenCtx.clearRect(0,0,hiddenCanvas.width,hiddenCanvas.height);
    ctx.clearRect(0,0,faceCanvas.width,faceCanvas.height);

        clearCanvases(hiddenCtx,ctx);

        const x = detection._box._x
        const y = detection._box._y
        const boxWidth = detection._box._width
        const boxHeight  = detection._box._height

hiddenCtx.drawImage(video,0,0,width,height);
ctx.drawImage(hiddenCanvas,x,y,boxWidth,boxHeight,0,0,IMG_WIDTH,IMG_HEIGHT);

}

function gotResults(err, result) {
    if (err) {
        console.log(err)
        return
    }

    if (result && result.length > 0) {
            drawInCanvas(result[0]);
    }

    else{
        clearCanvases(document.getElementById('hiddenCanvas').getContext('2d'),document.getElementById('faceCanvas').getContext('2d'));
    }

    if(detectionStarted)
    faceapi.detect(gotResults)
    
}


// Helper Functions

async function initVideoStream(){

    var videoElement = document.getElementById(videoSrcId);
    videoElement.setAttribute("style", "display: block"); 

    // Create a webcam capture
    const capture = await navigator.mediaDevices.getUserMedia({ video: true })
    videoElement.srcObject = capture;

}

function getVideoStream(){
    return document.getElementById(videoSrcId);
}

function showTrainingSet(trainingSet){

    let divElement = document.getElementById(trainingContainerId);
    divElement.innerHTML = "";
    convertToGreyscale(trainingSet);

    trainingData = trainingData.concat(trainingSet);

    document.getElementById('tCount').innerText = trainingData.length;

    for(let i = 0;i<trainingData.length;i++){
        let canvasElem = document.createElement('canvas');
        divElement.appendChild(canvasElem);
        canvasElem.width = trainingData[i].width;
        canvasElem.height = trainingData[i].height;
        canvasElem.getContext('2d').putImageData(trainingData[i].imgData,0,0);
    }
    
}

function toggleTrainingSet(){

    let bttn = document.getElementById('trainingSet');
    let emotion = document.getElementById('happy').checked ? 'Happy' : document.getElementById('sad').checked ? 'Sad' : 'Neutral';
    let freq = document.getElementById('oneSecFreq').checked ? 0.5 : document.getElementById('twoSecFreq').checked ? 1 : 2; 

    if(!trainingSetStarted){
        if(detectionStarted){
            currentTrainingSet = [];  // Clear Training Set
            bttn.innerText = 'Stop Training Set';
            trainingSetStarted = true;
            trainingSetIntervalId = setInterval(()=>{
            
            if(currentTrainingSet.length === BATCH_SIZE){
                window.clearInterval(trainingSetIntervalId);
                bttn.innerText = 'Create Training Set';
                trainingSetStarted = false;
                console.log(BATCH_SIZE+ ' Images Collected for Training');
                showTrainingSet(currentTrainingSet);
            }
            else{
                let canvas = document.getElementById('faceCanvas');
                let ctx = canvas.getContext('2d');
                currentTrainingSet.push(new TrainingImage(ctx.getImageData(0,0,canvas.width,canvas.height),canvas.width,canvas.height,emotion));
                console.log('Image Added to Training Set : '+currentTrainingSet.length);
            }
                
            },freq*1000)
        }
        else{
            console.log('Cannot Make Training Set. Please Start Detection First');
        }
    }

    else{
        trainingSetStarted = false;
        bttn.innerText = 'Create Training Set';
        window.clearInterval(trainingSetIntervalId);
        console.log(currentTrainingSet.length+' Images Collected for Training');
        showTrainingSet(currentTrainingSet);
    }

}

function convertToGreyscale(trainingSet){

    for(let index = 0;index<trainingSet.length;index++){
        var imgData = trainingSet[index].imgData;
        for(let i=0;i<imgData.data.length;i+=4){
            let avg = (imgData.data[i] + imgData.data[i+1] + imgData.data[i+2])/3;
            imgData.data[i] = avg;
            imgData.data[i+1] = avg;
            imgData.data[i+2] = avg;
        }

    }

}

function getLabelData(label){

let returnArr = [];

for(var i=0;i<label.length;i++){
returnArr.push(label[i] == 'Happy' ? 0 : label[i] == 'Sad' ? 1 : 2);
}

return tf.oneHot(tf.tensor1d(returnArr, 'int32'), classNames.length).toFloat();

}


function getImageAndLabel(size){

    var imgArr = [];
    var label = [];
    var trainSetSize = trainingData.length;
    for(let i=0;i<size;i++){
        let index = Math.floor(Math.random() * (trainSetSize-i));
        let tmp = trainingData[index];
        trainingData[index] = trainingData[trainSetSize-1-i];
        trainingData[trainSetSize-1-i] = tmp;
        imgArr.push(tmp);
    }

    for(let i=0;i<imgArr.length;i++){
        label.push(imgArr[i].emotion);
    }

    imgArr = tf.concat(getTensorArrayFromImageSet(imgArr));

    label = getLabelData(label);

    return [imgArr,label];
}



/*

Tensorflow Model Methods

*/

async function downloadModel(){

    await globalModelRef.save('downloads://faceDetect-js-model');
}

function getTensorArrayFromImageSet(trainingSet){
    var tensorArray = [];
    for(let index = 0;index<trainingSet.length;index++){
        tensorArray.push(tf.browser.fromPixels(trainingSet[index].imgData,1).expandDims());
    }
    return tensorArray;
}

async function startTraining(){

if(trainingData.length < 2 * BATCH_SIZE){
    alert('You need more training data to start training');
    return;
}

    document.getElementById('trainStatus').innerText = "In Progress";
    tfvis.show.modelSummary({name: 'Model Architecture'}, globalModelRef);  

    await trainModel(globalModelRef);

    }
    
async function startLiveFaceDetection(){
    await showAccuracy(globalModelRef, data);
    await showConfusion(globalModelRef, data);
    }


// Helper Methods

function makeTFModel() {

    const model = tf.sequential();
    
    const IMAGE_WIDTH = 50;
    const IMAGE_HEIGHT = 50;
    const IMAGE_CHANNELS = 1;  
    

    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 4,
      filters: 6,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
  

    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    model.add(tf.layers.conv2d({
      kernelSize: 4,
      filters: 12,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    

    model.add(tf.layers.flatten());
    
    const NUM_OUTPUT_CLASSES = 3;

   model.add(tf.layers.dropout({rate: 0.25}));

   model.add(tf.layers.dense({units: 256, activation: 'relu'}));

   model.add(tf.layers.dropout({rate: 0.5}));

   model.add(tf.layers.dense({units: NUM_OUTPUT_CLASSES, kernelInitializer: 'varianceScaling', activation: 'softmax'}));
  

    
    const optimizer = tf.train.adam();

    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    return model;
  }


async function trainModel(model) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const TRAIN_DATA_SIZE = parseInt(0.7*trainingData.length);
    const TEST_DATA_SIZE = trainingData.length - TRAIN_DATA_SIZE;
    


    const [trainXs, trainYs] = getImageAndLabel(TRAIN_DATA_SIZE);
  
    const [testXs, testYs] = getImageAndLabel(TEST_DATA_SIZE);
  
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 100,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

// Predictor Methods

async function importModel(){

    let modelName = document.getElementById('modelName').value;
    if(modelName == "" || modelName == null){
        alert('Please enter model name');
        return;
    }
    try{
    modelName = modelName+".json";
    globalModelRef = await tf.loadLayersModel('http://localhost:9090/mlapp/js/models/'+modelName);
    alert('Model imported successfully');

    const optimizer = tf.train.adam();
    globalModelRef.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
      });

    modelLoaded = true;
    }
    catch(e){
        console.log(e);
        alert('Please enter valid name');
        return;
    }

}

  function doPrediction(imgData) {

    for(let i=0;i<imgData.data.length;i+=4){
        let avg = (imgData.data[i] + imgData.data[i+1] + imgData.data[i+2])/3;
        imgData.data[i] = avg;
        imgData.data[i+1] = avg;
        imgData.data[i+2] = avg;
    }

    let ctx = document.getElementById('predictionCanvas').getContext('2d');
    ctx.clearRect(0,0,faceCanvas.width,faceCanvas.height);
    ctx.putImageData(imgData,0,0);
    const imgTensor = tf.browser.fromPixels(imgData,1).expandDims();
    const preds = globalModelRef.predict(imgTensor);
  
    imgTensor.dispose();
    return preds;

  }
  
  function predict(){

    if(!modelLoaded){
        alert('Model not loaded');
        return;
    }

    if(predictionTimer == null){
        document.getElementById('predict').innerText = 'Stop Prediction';
    }

    else{
        window.clearInterval(predictionTimer);
        document.getElementById('predict').innerText = 'Start Prediction';
        return;
    }

    var emotionSpan = document.getElementById('emotion');

    predictionTimer = setInterval(()=>{
    let canvas = document.getElementById('faceCanvas');
                let ctx = canvas.getContext('2d');
                let prediction = doPrediction(ctx.getImageData(0,0,canvas.width,canvas.height));
                console.log(prediction.print());
                const em = prediction.argMax(-1).arraySync();
                emotionSpan.innerText = em == 0 ? 'Happy' : em == 1 ? 'Sad' : 'Neutral';
    },1000);
  }

