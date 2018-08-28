# Purely functional ML Framework for Scala 

Right now it is just a basic concept. You are welcome to contribute
to bring ML into the Scala ecosystem.

There is an example of training ANN on 
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset:

```scala
  "neural network" should "classify MNIST data set" in {
    val training = MNIST.loadTrainingSet(2000)
    val test = MNIST.loadTestSet(100)
    val (factor, input) = normalize(training.input)
    val model = Dense(25, Sigmoid(), kernelReg = L2(0.1)) |&| Dense(10, Sigmoid(), kernelReg = L2(0.1))
    val weights = Adam(rate = 0.02, batch = 2000)
      .minimize(nnError(model, training.labels), input)
      .through(iter(500))
      .observe(logStdOut)
      .observe(plotToFile("MNIST.png"))
      .runSync.vars
    val classifier = nn(model)(weights)
    val prediction = classifier(normalize(test.input, factor))
    val accuracy = binaryAccuracy(test.labels, prediction)
    accuracy should be > 0.7
  }
```


### todo list:

- Stochastic Gradient Decent (SGD) -- DONE --
- SDG features (momentum, nesterow) -- DONE --
- Advanced Gradient Decent optimizations (AdaGrad, AdaDelta, RMSProp, Adam, NAdam, Adamax) -- DONE --
- Linear regression -- DONE --
- Logistic regression -- DONE --
- Simple ANN (Artificial Neural Network) -- DONE (improvements...) --
- Advanced ANN -- TODO --
- REST service for Simple ANN -- TODO --
- UI for Simple ANN -- TODO --
- Simple Recurrent Neural Network -- TODO --
- -- TODO --


