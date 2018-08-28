package io.scanet.core

import breeze.linalg.DenseVector.zeros
import breeze.linalg.{DenseMatrix, DenseVector}
import io.scanet.syntax._
import io.scanet.core.func._
import io.scanet.test.CustomMatchers
import org.scalatest.FlatSpec

class FuncTest extends FlatSpec with CustomMatchers {

  "linear function x1 + x2" should "have right result" in {
    Linear(DenseVector(1.0, 1.0))(DenseVector(1.0, 2.0)) should be(3.0)
  }

  it should "have right gradient" in {
    Linear(DenseVector(1.0, 1.0)) gradient DenseVector(1.0, 2.0) should be(DenseVector(1.0, 1.0))
  }

  "linear regression function" should "have valid implementation" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    linearRegression(coef)(zeros[Double](2)) should beWithinTolerance(32.07, 0.01)
  }

  it should "have right gradient" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    linearRegression(coef) gradient zeros[Double](2) should beWithinTolerance(DenseVector(-5.83, -65.32), 0.01)
  }

  "logistic regression function" should "have valid implementation" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    logisticRegression(coef)(zeros[Double](3)) should beWithinTolerance(0.693, 0.01)
  }

  it should "have right gradient" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    logisticRegression(coef) gradient zeros[Double](3) should beWithinTolerance(DenseVector(-0.1, -12.0092, -11.2628), 0.01)
  }

  "function combining" should "work" in {
    val c = Linear(DenseVector(1.0, 1.0)) |&| Linear(DenseVector(2.0, 2.0))
    c(DenseVector(1.0, 2.0)) should be(9.0)
  }

  "builder combining" should "work" in {
    val b1: DFBuilder[Linear] = coef => Linear(coef(0, ::).t)
    val b2: DFBuilder[Linear] = coef => Linear(coef(0, ::).t + 1.0)
    val b12: DFBuilder[(Linear, Linear)] = b1 <&> b2
    b12(DenseMatrix((1.0, 2.0)))(DenseVector(1.0, 1.0)) should be(8.0)
  }

  "sigmoid function" should "have right result" in {
    Sigmoid() apply1 0.0 should beWithinTolerance(0.5, 0.01)
  }

  it should "have right result when bulk calculation" in {
    Sigmoid() apply1 DenseVector(0.0, 1.0, 10.0) should beWithinTolerance(DenseVector(0.5, 0.731, 0.999), 0.01)
  }

  it should "have right gradient" in {
    Sigmoid() gradient1 0.0 should beWithinTolerance(0.25, 0.01)
  }

  it should "have right gradient when bulk calculation" in {
    Sigmoid() gradient1 DenseVector(0.0, 1.0, 5) should beWithinTolerance(DenseVector(0.25, 0.196, 0), 0.01)
  }
}
