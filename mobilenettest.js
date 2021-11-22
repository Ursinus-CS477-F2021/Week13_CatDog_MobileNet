let model;
let image;
tf.loadLayersModel(
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json').then(
  m => {
  model = m;
  image = new Image();
  image.src = "Theo.png";
  image.onload = function() {
    model.classify(image).then(predictions => {
      console.log('Predictions: ');
      console.log(predictions);
    });
  }
});
