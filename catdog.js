// Some inspiration from
// https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/mnist-core/data.js


/**
 * 
 * @param {string} prefix File prefix
 * @param {int} res Square resolution to which to resize all images
 * @param {int} i1 Start index of image to load
 * @param {int} i2 End index of image to load
 * @param {int} i Index of current image
 * @param {array} images List of images that are being built, as Float32Arrays
 */
function loadFiles(prefix, res, i1, i2, i, images) {
  const progressArea = document.getElementById("progress");
  return new Promise(resolve => {
    if (i === undefined) {
      i = i1;
      images = [];
    }
    let path = prefix + "." + i + ".jpg";
    progressArea.innerHTML = "Loading " + path;
    loadImage(path, res).then((buffi) => {
      images.push(buffi);
      if (i+1 <= i2) {
        loadFiles(prefix, res, i1, i2, i+1, images).then(()=>{resolve(images)});
      }
      else {
        resolve(images);
      }
    }) 
  });
}


class CatDog {
  constructor(imgRes, NTrain, NTest) {
    this.imgRes = imgRes;
    this.NTrain = NTrain;
    this.NTest = NTest;
    this.initModel();
    this.xs = [];
    this.ys = [];
    const that = this;
    this.loadData().then(()=>{
      this.trainIndices = tf.util.createShuffledIndices(NTrain);
      this.testIndices = tf.util.createShuffledIndices(NTest);
      this.xstrain = this.xs.slice(0, NTrain*imgRes*imgRes*3);
      this.ystrain = this.ys.slice(0, NTrain);
      this.xstest = this.xs.slice(NTrain*imgRes*imgRes*3);
      this.ystest = this.ys.slice(NTrain); 
      that.trainModel();
    });
  }

  initModel() {
    const model = tf.sequential();
    
    // In the first layer of our convolutional neural network we have 
    // to specify the input shape. Then we specify some parameters for 
    // the convolution operation that takes place in this layer.
    model.add(tf.layers.conv2d({
      inputShape: [this.imgRes, this.imgRes, 3],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
  
    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.  
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Repeat another conv2d + maxPooling stack. 
    // Note that we have more filters in the convolution.
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten());
  
    // Our last layer is a dense layer which has 1 output unit
    model.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid'
    }));
  
  
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.sigmoidCrossEntropy,
      metrics: ['accuracy'],
    });
    
    this.model = model;
  }

  loadData() {
    const that = this;
    return new Promise(resolve => {
      let N = that.NTrain + that.NTest;
      const res = that.imgRes;
      loadFiles("train/cat", res, 0, N-1).then(cat => {
        loadFiles("train/dog", res, 0, N-1).then(dog => {
          let xs = new Float32Array(N*2*res*res*3);
          let ys = new Float32Array(N*2);
          for (let i = 0; i < N; i++) {
            // Interleave cats and dogs to make train/test split easier
            xs.set(cat[i], i*2*res*res*3);
            ys[i*2] = 0;
            xs.set(dog[i], (i*2+1)*res*res*3);
            ys[i*2+1] = 1;
          }
          that.xs = xs;
          that.ys = ys;
          resolve();
        });
      });
    });
  }

  trainModel() {
    const that = this;
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const [trainXs, trainYs] = tf.tidy(() => {
      return that.nextTrainBatch(this.NTrain);
    });
  
    const [testXs, testYs] = tf.tidy(() => {
      return that.nextTrainBatch(this.NTest);
    });
  
    return this.model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });  
  }


  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize, [this.xstrain, this.ystrain], () => {
        this.shuffledTrainIndex =
            (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.xstest, this.ystest], () => {
      this.shuffledTestIndex =
          (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const res = this.imgRes;
    let xs = new Float32Array(batchSize*res*res*3);
    let ys = new Uint8Array(batchSize);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();
      const image = data[0].slice(idx*res*res*3, (idx+1)*res*res*3);
      xs.set(image, i*res*res*3);
      ys[i] = data[1][i];
    }
    xs = tf.tensor(xs, [batchSize, res, res, 3]);
    ys = tf.tensor(ys);
    return [xs, ys];
  }
}


let cd = new CatDog(128, 200, 100);