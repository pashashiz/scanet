package io.scanet.nn

import breeze.linalg._
import io.scanet.func.{Sigmoid, SigmoidInst}
import io.scanet.optimizers.{Adam, SGD}
import io.scanet.syntax._
import io.scanet.test.CustomMatchers
import org.scalatest.FlatSpec

import scala.concurrent.ExecutionContext.Implicits._


class NNTest extends FlatSpec with CustomMatchers with DenseLayerInst with OtherLayersInst with SigmoidInst with NNInst {

  "dense layer" should "have working forward propagation" in {
    // M: 4, IN: 3, OUT: 4
    val input = DenseMatrix(
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 1.0, 1.0))
    val coef = DenseMatrix(
      (1.0, 0.1, 1.0),
      (0.5, 1.0, 0.0),
      (1.0, 1.0, 0.2),
      (0.1, 1.0, 0.3))
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
      (1.0, 0.1, 1.0),
      (0.5, 1.0, 0.0),
      (1.0, 1.0, 0.2),
      (0.1, 1.0, 0.3))
    val error = DenseMatrix(
      (-0.0146, -0.0732, -0.1465, 0.000),
      (0.0041, 0.0203, 0.0407, 0.000),
      (0.0043, 0.0215, 0.0430, 0.000),
      (-0.0127, -0.0637, -0.1274, 0.000))
    val (delta, grad) = Dense(4, activation = Sigmoid()) backprop (List(coef), input, error)
    val expected = DenseMatrix(
      (-0.007, -0.004, -0.003),
      (-0.004, -0.005, -0.019),
      (-0.003, -0.004, -0.032),
      (-0.000, -0.000, -0.000))
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
      (1.0, 0.1, 1.0),
      (0.5, 1.0, 0.0),
      (1.0, 1.0, 0.2),
      (0.1, 1.0, 0.3))
    val coef2 = DenseMatrix(
      (0.1, 0.5, 1.0, 0.0))
    val coef = List(coef1, coef2)
    val nn = Dense(4, Sigmoid()) |+| Dense(1, Sigmoid())
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
      (1.0, 0.1, 1.0),
      (0.5, 1.0, 0.0),
      (1.0, 1.0, 0.2),
      (0.1, 1.0, 0.3))
    val coef2 = DenseMatrix(
      (0.1, 0.5, 1.0, 0.0))
    val coef = List(coef1, coef2)
    val nn = Dense(4, Sigmoid()) |+| Dense(1, Sigmoid())
    val output = nn forward (coef, input)
    val outputExpected = DenseMatrix(
      0.0,
      1.0,
      1.0,
      0.0)
    val error = outputExpected - output
    val (delta, grad) = nn backprop (coef, input, error)
    val grad1::grad2::Nil = grad
    val expectedGrad1 = DenseMatrix(
      (-0.007, -0.004, 0.003),
      (-0.004, -0.005, -0.019),
      (-0.003, -0.004, -0.032),
      (-0.000, -0.000, -0.000))
    grad1 should beWithinTolerance(expectedGrad1, 0.01)
    val expectedGrad2 = DenseMatrix(
      (-0.152, -0.121, -0.131, -0.129))
    grad2 should beWithinTolerance(expectedGrad2, 0.01)
  }

  "simple neural network" should "work" in {
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
    val coef1 = DenseMatrix.rand[Double](4, 3) - 0.5
    val coef2 = DenseMatrix.rand[Double](1, 4) - 0.5
    val coef = List(coef1, coef2)
    val layers = Dense(4, Sigmoid()) |+| Dense(1, Sigmoid())
    val theta = Adam(rate = 0.3)
      .minimize(nnError(layers, output), input, layers.pack(coef))
      .through(iter(500))
      .observe(logStdOut)
      .observe(plotToFile("Adam:simple-ANN.png"))
      .runSync.vars
    var func = nnError(layers, output).apply(input)
    func(theta) should beWithinTolerance(0, 0.1)
  }

  "neural network instead of logistic regression" should "work" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val (scale, input) = normalize(read(::, 0 to 1))
    val output = read(::, 2).toDenseMatrix.t
    val coef1 = DenseMatrix.rand[Double](4, 2) - 0.5
    val coef2 = DenseMatrix.rand[Double](1, 4) - 0.5
    val coef = List(coef1, coef2)
    val model = Dense(4, Sigmoid()) |+| Dense(1, Sigmoid())
    val theta = SGD(rate = 0.5)
      .minimize(nnError(model, output), input, model.pack(coef))
      .through(iter(50))
      .observe(logStdOut)
      .observe(plotToFile("Adam:ANN-instead-of-logistic.png"))
      .runSync.vars
    val classifier = nn(model)(theta)
    val prediction = classifier(input).map(value => if (value > 0.5) 1.0 else 0.0)
    val accuracy = sum((prediction :== output).map(value => if (value) 1.0 else 0.0))
    accuracy should beWithinTolerance(91, 5)
  }

}
