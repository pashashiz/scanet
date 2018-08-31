package io.scanet.nn

import breeze.linalg.DenseMatrix.horzcat
import breeze.linalg._
import io.scanet.core.func._
import io.scanet.optimizers.{Adam, SGD}
import io.scanet.syntax._
import io.scanet.test.CustomMatchers
import org.scalatest.FlatSpec

import scala.concurrent.ExecutionContext.Implicits._


class CombinedLayerSpec extends FlatSpec with CustomMatchers {

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
}
