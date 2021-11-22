const osImgCanvas = document.createElement('canvas');
const osImgCtx = osImgCanvas.getContext('2d');


/**
 * Load an image from a path and resize it to a particular resolution
 * @param {string} path Path to image file
 * @param {int} res Square resolution to which to resize image
 * 
 * Resolves {Float32Array} buffer Pointer to memory to which to write this image
 */
 function loadImage(path, res) {
    return new Promise( (resolve, reject) => {
        osImgCanvas.width = res;
        osImgCanvas.height = res;
        let buffer = new Float32Array(res*res*3);
        image = new Image();
        image.src = path;
        image.onload = function() {
            osImgCtx.clearRect(0, 0, res, res);
            osImgCtx.drawImage(image, 0, 0, image.width, image.height, 0, 0, res, res);
            const imageData = osImgCtx.getImageData(0, 0, res, res);
            for (let i = 0; i < imageData.data.length/4; i++) {
                for (let k = 0; k < 3; k++) {
                    buffer[i*3+k] = imageData.data[i*4+k]/255.0;
                }
            }
            resolve(buffer);
        }
    });
  }
  