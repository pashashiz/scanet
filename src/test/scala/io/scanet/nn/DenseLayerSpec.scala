package io.scanet.nn

import breeze.linalg._
import io.scanet.core.func._
import io.scanet.syntax._
import io.scanet.test.CustomMatchers
import org.scalatest.FlatSpec

class DenseLayerSpec extends FlatSpec with CustomMatchers {

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

  it should "have working forward propagation penalty" in {
    val coef = DenseMatrix(
      (0.0, 1.0, 0.1, 1.0),
      (0.0, 0.5, 1.0, 0.0),
      (0.0, 1.0, 1.0, 0.2),
      (0.0, 0.1, 1.0, 0.3))
    val penalty = Dense(4, activation = Sigmoid(), kernelReg = L2(0.25)) forwardPenalty List(coef)
    penalty should be (0.8)
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

  it should "have working back propagation penalty" in {
    val coef = DenseMatrix(
      (0.0, 1.0, 0.1, 1.0),
      (0.0, 0.5, 1.0, 0.0),
      (0.0, 1.0, 1.0, 0.2),
      (0.0, 0.1, 1.0, 0.3))
    val penalty = Dense(4, activation = Sigmoid(), kernelReg = L2(0.25)) backwardPenalty  List(coef)
    val expected = DenseMatrix(
      (0.000, 0.250, 0.025, 0.250),
      (0.000, 0.125, 0.250, 0.000),
      (0.000, 0.250, 0.250, 0.050),
      (0.000, 0.025, 0.250, 0.075))
    penalty.head should beWithinTolerance(expected, 0.01)
  }
}
