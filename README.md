# Purely functional ML Framework for Scala 

Right now it is just a basic concept. You are welcome to contribute
to bring ML into the Scala ecosystem.

## Examples

### ANN

There is an example of training ANN on 
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset:

```scala
"neural network" should "classify MNIST data set" in {
    val training = MNIST.loadTrainingSet(10000)
    val test = MNIST.loadTestSet(100)
    val (factor, input) = normalize(training.input)
    val model = Dense(50, Sigmoid(), kernelReg = L2(0.005)) |&| Dense(10, Sigmoid(), kernelReg = L2(0.005))
    val weights = Adam(rate = 0.01, batch = 100)
      .minimize(nnError(model), horzcat(input, training.labels))
      .through(epoch(10))
      .observe(logStdOut)
      .observe(plotToFile("MNIST.png"))
      .runSync.vars
    val classifier = nn(model)(weights)
    val prediction = classifier(normalize(test.input, factor))
    val accuracy = binaryAccuracy(test.labels, prediction)
    accuracy should be > 0.9
  }
```


## Features:

- [x] Stochastic Gradient Decent (SGD)
- [x] SDG features (momentum, nesterow)
- [x] Advanced Gradient Decent optimizations (AdaGrad, AdaDelta, RMSProp, Adam, NAdam, Adamax)
- [x] Linear regression
- [x] Logistic regression
- [x] Artificial NN (Neural Network)
- [ ] Performance
- [ ] Documentation 
- [ ] Convolutional NN
- [ ] Metrics
- [ ] Preprocessors
- [ ] Visualize NN learning process
  - REST service
  - UI 
- [ ] Recurrent NN
- [ ] Tensor Flow as a backend

