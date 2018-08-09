package io.scanet.optimizers

import breeze.linalg._
import io.scanet.func._
import io.scanet.syntax._
import io.scanet.test.CustomMatchers
import org.scalatest.FlatSpec

import scala.concurrent.ExecutionContext.Implicits._

class OptimizeLogisticRegressionTest extends FlatSpec with CustomMatchers {

  "SGD with momentum" should "find a minimum of [logistic regression] within 1000 epochs" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = SGD(25, 0.002, 0.99)
      .minimize(logisticRegression, coef, DenseVector(0, 0, 0))
      .through(epoch(1000))
      .observe(logStdOut)
      .observe(plotToFile("Momentum:logistic.png"))
      .runSync.vars
    logisticRegression(coef)(vars) should beWithinTolerance(0.203, 1)
  }

  "SGD with nesterov momentum" should "find a minimum of [logistic regression] within 100000 epochs" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = SGD(25, 0.001, 0.6, nesterov = true)
      .minimize(logisticRegression, coef, DenseVector(0, 0, 0))
      .through(epoch(100000))
      .observe(logStdOut)
      .observe(plotToFile("Nesterov:logistic.png"))
      .runSync.vars
    logisticRegression(coef)(vars) should beWithinTolerance(0.203, 1)
  }

  "AdaGrad" should "find a minimum of [logistic regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = AdaGrad(25, 1)
      .minimize(logisticRegression, coef, DenseVector(0, 0, 0))
      .through(epoch(5000))
      .observe(logStdOut)
      .observe(plotToFile("AdaGrad:logistic.png"))
      .runSync.vars
    logisticRegression(coef)(vars) should beWithinTolerance(0.203, 1)
  }

  "AdaDelta" should "find a minimum of [logistic regression] within 250 epochs" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = AdaDelta(25, 1, 0.9)
      .minimize(logisticRegression, coef, DenseVector(0, 0, 0))
      .through(epoch(15000))
      .observe(logStdOut)
      .observe(plotToFile("AdaDelta:logistic.png"))
      .runSync.vars
    logisticRegression(coef)(vars) should beWithinTolerance(0.203, 1)
  }

  "RMSProp" should "find a minimum of [logistic regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = RMSProp(20, 0.01, 0.9)
      .minimize(logisticRegression, coef, DenseVector(0, 0, 0))
      .through(epoch(5000))
      .observe(logStdOut)
      .observe(plotToFile("RMSProp:logistic.png"))
      .runSync.vars
    logisticRegression(coef)(vars) should beWithinTolerance(0.203, 1)
  }

  "Adam" should "find a minimum of [logistic regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = Adam(20, 0.1, 0.9, 0.999)
      .minimize(logisticRegression, coef, DenseVector(0, 0, 0))
      .through(epoch(1000))
      .observe(logStdOut)
      .observe(plotToFile("Adam:logistic.png"))
      .runSync.vars
    logisticRegression(coef)(vars) should beWithinTolerance(0.203, 1)
  }

  "Adamax" should "find a minimum of [logistic regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = Adamax(20, 0.0001, 0.9, 0.999)
      .minimize(logisticRegression, coef, DenseVector(0, 0, 0))
      .through(epoch(2000))
      .observe(logStdOut)
      .observe(plotToFile("Adamax:logistic.png"))
      .runSync.vars
    logisticRegression(coef)(vars) should beWithinTolerance(0.203, 1)
  }

  "Nadam" should "find a minimum of [logistic regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("logistic_regression_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = Nadam(20, 0.03, 0.9, 0.999)
      .minimize(logisticRegression, coef, DenseVector(0, 0, 0))
      .through(epoch(3000))
      .observe(logStdOut)
      .observe(plotToFile("Nadam:logistic.png"))
      .runSync.vars
    logisticRegression(coef)(vars) should beWithinTolerance(0.203, 1)
  }
}