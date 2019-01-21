let resolution = 50;
let cols;
let rows;
let xs;
let training = false;

const train_xs = tf.tensor2d([
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1]
]);

const train_ys = tf.tensor2d([
  [0],
  [1],
  [1],
  [0]
]);

function setup() {
  createCanvas(400, 400);

  cols = width/resolution;
  rows = height/resolution;

  let inputs = [];
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      let x1 = i / cols;
      let x2 = j / rows;
      inputs.push([x1, x2]);
    }
  }
  xs = tf.tensor2d(inputs);
}

async function trainModel() {
  training = true;
  return await model.fit(train_xs, train_ys, {
    shuffle: true,
    epochs: 10,
  });
}

function draw() {
  background(0);

  tf.tidy(() => {
    if (training == false) {
      trainModel().then(res => {
        console.log(res.history.loss[0])
        training = false;
      });
    }
  });

  tf.tidy(() => {
    let ys = model.predict(xs)
    let y_values = ys.dataSync();

    let index = 0;
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        let br = y_values[index] * 255
        fill(br);
        rect(i * resolution, j * resolution, resolution, resolution);
        fill(255);
        textAlign(CENTER);
        text(nf(y_values[index], 1, 2), i * resolution + resolution/2, j * resolution + resolution/2);
        index++;
      }
    }
  });

}
