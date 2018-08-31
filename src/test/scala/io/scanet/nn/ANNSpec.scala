package io.scanet.nn

import breeze.linalg.DenseMatrix.horzcat
import breeze.linalg._
import io.scanet.core.func._
import io.scanet.core.metrics.binaryAccuracy
import io.scanet.linalg.splitColsAt
import io.scanet.optimizers.{Adam, SGD}
import io.scanet.preprocessing.MNIST
import io.scanet.syntax._
import io.scanet.test.CustomMatchers
import org.scalatest.FlatSpec

import scala.concurrent.ExecutionContext.Implicits._


class ANNSpec extends FlatSpec with CustomMatchers {

  "simple artificial neural network" should "be optimized with low error" in {
    // M: 4, IN: 3, Layer 1: 4, Layer 2: 1
    val input = DenseMatrix(
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 1.0, 1.0))
    val output = DenseMatrix(
      0.0,
      1.0,
      1.0,
      0.0)
    val layers = Dense(4, Sigmoid()) |&| Dense(1, Sigmoid())
    val weights = Adam(rate = 0.3)
      .minimize(nnError(layers), horzcat(input, output))
      .through(iter(500))
      .observe(logStdOut)
      .observe(plotToFile("Adam:simple-ANN.png"))
      .runSync.vars
    var error = nnError(layers).apply(horzcat(input, output))
    error(weights) should beWithinTolerance(0, 0.1)
  }

  it should "classify with high accuracy" in {
    val read = csvread(resource("logistic_regression_1.scv"))
    val (inputRaw, output) = splitColsAt(read, 2)
    val (_, input) = normalize(inputRaw)
    val (learning, training) = (0 to 89, 90 to 99)
    val model = Dense(4, Sigmoid()) |&| Dense(1, Sigmoid())
    val weights = SGD(rate = 0.5)
      .minimize(nnError(model), horzcat(input(learning, ::), output(learning, ::)))
      .through(iter(50))
      .observe(logStdOut)
      .observe(plotToFile("Adam:ANN-instead-of-logistic.png"))
      .runSync.vars
    val classifier = nn(model)(weights)
    val prediction = classifier(input(training, ::))
    binaryAccuracy(output(training, ::), prediction) should be > 0.9
  }

  /**
    * | Tweak | H layer |   N   | Epochs | Batch | Optimizer | L rate |  Reg  | beta1 | beta2 | Converged | Accuracy |
    * ----------------------------------------------------------------------------------------------------------------
    * |  Reg  |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.002 |  0.9  | 0.99  |     +     |   0.81   |
    * |  Reg  |    25   | 10000 |   10   |  100  |   Adam    |  0.01  |<0.005>|  0.9  | 0.99  |     +     |   0.87   |
    * |  Reg  |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.01  |  0.9  | 0.99  |     +     |   0.86   |
    * |  Reg  |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.02  |  0.9  | 0.99  |     +     |   0.84   |
    * |  LR   |    25   | 10000 |   10   |  100  |   Adam    |  0.02  | 0.005 |  0.9  | 0.99  |     +     |   0.85   |
    * |  LR   |    25   | 10000 |   10   |  100  |   Adam    | <0.01> | 0.005 |  0.9  | 0.99  |     +     |   0.87   |
    * |  LR   |    25   | 10000 |   25   |  100  |   Adam    |  0.005 | 0.005 |  0.9  | 0.99  |     +     |   0.86   |
    * |  B1   |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.005 |  0.5  | 0.99  |     +     |   0.86   |
    * |  B1   |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.005 | <0.9> | 0.99  |     +     |   0.87   |
    * |  B1   |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.005 |  0.95 | 0.99  |     +     |   0.83   |
    * |  B2   |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.005 |  0.9  | 0.9   |     +     |   0.82   |
    * |  B2   |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.005 |  0.9  |<0.99> |     +     |   0.87   |
    * |  B2   |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.005 |  0.9  | 0.999 |     +     |   0.86   |
    * | Batch |    25   | 10000 |   10   |   50  |   Adam    |  0.01  | 0.005 |  0.9  | 0.99  |     +     |   0.86   |
    * | Batch |    25   | 10000 |   10   | <100> |   Adam    |  0.01  | 0.005 |  0.9  | 0.99  |     +     |   0.87   |
    * | Batch |    25   | 10000 |   30   |  200  |   Adam    |  0.01  | 0.005 |  0.9  | 0.99  |     +     |   0.86   |
    * | Batch |    25   | 10000 |   50   |  500  |   Adam    |  0.01  | 0.005 |  0.9  | 0.99  |     +     |   0.86   |
    * |   N   |    25   |  5000 |   20   |  100  |   Adam    |  0.01  | 0.005 |  0.9  | 0.99  |     +     |   0.79   |
    * |   N   |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.005 |  0.9  | 0.99  |     +     |   0.87   |
    * |   N   |    25   | 20000 |    6   |  100  |   Adam    |  0.01  | 0.005 |  0.9  | 0.99  |     +     |   0.90   |
    * |   N   |    25   |<60000>|    3   |  100  |   Adam    |  0.01  | 0.005 |  0.9  | 0.99  |     +     |   0.90   |
    * |   HL  |    25   | 10000 |   10   |  100  |   Adam    |  0.01  | 0.005 |  0.9  | 0.99  |     +     |   0.87   |
    * |   HL  |   <50>  | 10000 |   20   |  100  |   Adam    |  0.01  | 0.005 |  0.9  | 0.99  |     +     |  <0.92>  |
    */
  "artificial neural network" should "classify MNIST data set" in {
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
    println(s"accuracy: $accuracy")
    accuracy should be > 0.9
  }
}
