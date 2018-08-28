package io.scanet.nn

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


class NNTest extends FlatSpec with CustomMatchers {

  "dense layer" should "have working forward propagation" in {
    // M: 4, IN: 3, OUT: 4
    val input = DenseMatrix(
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 1.0, 1.0))
    val coef = DenseMatrix(
      (0.0, 1.0, 0.1, 1.0),
      (0.0, 0.5, 1.0, 0.0),
      (0.0, 1.0, 1.0, 0.2),
      (0.0, 0.1, 1.0, 0.3))
    val output = Dense(4, activation = Sigmoid()) forward (List(coef), input)
    val expected = DenseMatrix(
      (0.731, 0.500, 0.549, 0.574),
      (0.750, 0.731, 0.768, 0.785),
      (0.880, 0.622, 0.768, 0.598),
      (0.890, 0.817, 0.900, 0.802))
    output should beWithinTolerance(expected, 0.01)
  }

  it should "have working back propagation" in {
    // M: 4, IN: 3, OUT: 4
    val input = DenseMatrix(
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 1.0, 1.0))
    val coef = DenseMatrix(
      (0.0, 1.0, 0.1, 1.0),
      (0.0, 0.5, 1.0, 0.0),
      (0.0, 1.0, 1.0, 0.2),
      (0.0, 0.1, 1.0, 0.3))
    val error = DenseMatrix(
      (-0.0146, -0.0732, -0.1465, 0.000),
      (0.0041, 0.0203, 0.0407, 0.000),
      (0.0043, 0.0215, 0.0430, 0.000),
      (-0.0127, -0.0637, -0.1274, 0.000))
    val (delta, grad) = Dense(4, activation = Sigmoid()) backward (List(coef), input, error)
    val expected = DenseMatrix(
      (-0.002, -0.007, -0.004, -0.003),
      (-0.019, -0.004, -0.005, -0.019),
      (-0.033, -0.003, -0.004, -0.032),
      (-0.000, -0.000, -0.000, -0.000))
    grad.head should beWithinTolerance(expected, 0.01)
  }

  "combined layer" should "have working forward propagation" in {
    // M: 4, IN: 3, Layer 1: 4, Layer 2: 1
    val input = DenseMatrix(
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 1.0, 1.0))
    val coef1 = DenseMatrix(
      (0.0, 1.0, 0.1, 1.0),
      (0.0, 0.5, 1.0, 0.0),
      (0.0, 1.0, 1.0, 0.2),
      (0.0, 0.1, 1.0, 0.3))
    val coef2 = DenseMatrix(
      (0.0, 0.1, 0.5, 1.0, 0.0))
    val coef = List(coef1, coef2)
    val nn = Dense(4, Sigmoid()) |&| Dense(1, Sigmoid())
    val output = nn forward (coef, input)
    val expected = DenseMatrix(
      0.705,
      0.770,
      0.762,
      0.801)
    output should beWithinTolerance(expected, 0.01)
  }

  it should "have working back propagation" in {
    // M: 4, IN: 3, Layer 1: 4, Layer 2: 1
    val input = DenseMatrix(
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 1.0, 1.0))
    val coef1 = DenseMatrix(
      (0.0, 1.0, 0.1, 1.0),
      (0.0, 0.5, 1.0, 0.0),
      (0.0, 1.0, 1.0, 0.2),
      (0.0, 0.1, 1.0, 0.3))
    val coef2 = DenseMatrix(
      (0.0, 0.1, 0.5, 1.0, 0.0))
    val coef = List(coef1, coef2)
    val nn = Dense(4, Sigmoid()) |&| Dense(1, Sigmoid())
    val output = nn forward (coef, input)
    val outputExpected = DenseMatrix(
      0.0,
      1.0,
      1.0,
      0.0)
    val error = outputExpected - output
    val (delta, grad) = nn backward (coef, input, error)
    val grad1::grad2::Nil = grad
    val expectedGrad1 = DenseMatrix(
      (-0.003, -0.001, -0.000, -0.003),
      (-0.018, -0.004, -0.005, -0.019),
      (-0.032, -0.004, -0.004, -0.032),
      ( 0.000,  0.000,  0.000,  0.000))
    grad1 should beWithinTolerance(expectedGrad1, 0.01)
    val expectedGrad2 = DenseMatrix(
      (-0.191, -0.152, -0.121, -0.131, -0.129))
    grad2 should beWithinTolerance(expectedGrad2, 0.01)
  }

  // TODO: test regularization

  "simple neural network" should "be optimized with low error" in {
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
      .minimize(nnError(layers, output), input)
      .through(iter(500))
      .observe(logStdOut)
      .observe(plotToFile("Adam:simple-ANN.png"))
      .runSync.vars
    var error = nnError(layers, output).apply(input)
    error(weights) should beWithinTolerance(0, 0.1)
  }

  "neural network" should "classify with high accuracy" in {
    val read = csvread(resource("logistic_regression_1.scv"))
    val (inputRaw, output) = splitColsAt(read, 2)
    val (_, input) = normalize(inputRaw)
    val (learning, training) = (0 to 89, 90 to 99)
    val model = Dense(4, Sigmoid()) |&| Dense(1, Sigmoid())
    val weights = SGD(rate = 0.5)
      .minimize(nnError(model, output(learning, ::)), input(learning, ::))
      .through(iter(50))
      .observe(logStdOut)
      .observe(plotToFile("Adam:ANN-instead-of-logistic.png"))
      .runSync.vars
    val classifier = nn(model)(weights)
    val prediction = classifier(input(training, ::))
    binaryAccuracy(output(training, ::), prediction) should be > 0.9
  }

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
    println(s"accuracy: $accuracy")
    accuracy should be > 0.7
  }

}
