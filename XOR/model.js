
const model = tf.sequential();

const hidden = tf.layers.dense({
  units: 2,
  inputShape: [2],
  activationFunction: "sigmoid"
});

model.add(hidden);

const output = tf.layers.dense({
  units: 1,
  activationFunction: "sigmoid"
})

model.add(output);

const optimizer = tf.train.sgd(0.1);
model.compile({optimizer: optimizer, loss: "meanSquaredError"});
