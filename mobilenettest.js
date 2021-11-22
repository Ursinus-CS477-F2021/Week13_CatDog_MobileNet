const imageArea = document.getElementById("imageArea");
const progressArea = document.getElementById("progressArea");
const predictionsArea = document.getElementById("predictionsArea");

let modelLoaded = false;
let model = mobilenet.load();
let image;
/*tf.loadLayersModel(
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json').then(
  m => {
  model = m;
});*/

function classifyImage(image) {
  if (!modelLoaded) {
    progressArea.innerHTML = "Loading mobilenet...";
    model.then(m => {
      model = m;
      modelLoaded = true;
      classifyImage(image);
    });
  }
  else {
    imageArea.innerHTML = "";
    imageArea.appendChild(image);
    progressArea.innerHTML = "Classifying image...";
    model.classify(image).then(predictions => {
      progressArea.innerHTML = "Finished predictions";
      console.log(predictions)
      predictionsArea.innerHTML = "";
      let table = document.createElement("table");
      for (let i = 0; i < predictions.length; i++) {
        let tr = document.createElement("tr");
        let td = document.createElement("td");
        td.innerHTML = predictions[i].className;
        tr.appendChild(td);
        td = document.createElement("td");
        td.innerHTML = predictions[i].probability;
        tr.appendChild(td);
        table.appendChild(tr);
      }
      predictionsArea.appendChild(table);
    });
  }
}


image = new Image();
image.src = "Theo.png";
image.onload = function() {
  classifyImage(image);
}


let imageInput = document.getElementById('imageInput');
imageInput.addEventListener('change', function(e) {
    let reader = new FileReader();
    reader.onload = function(e) {
        let arrayBufferView = new Uint8Array(e.target.result);
        let blob = new Blob([arrayBufferView], {type: imageInput.files[0].type});
        let urlCreator = window.URL || window.webkitURL;
        let imageUrl = urlCreator.createObjectURL(blob);
        let image = new Image();
        image.src = imageUrl;
        image.onload = function() {
            classifyImage(image);
        }
    }
    reader.readAsArrayBuffer(imageInput.files[0]);
});