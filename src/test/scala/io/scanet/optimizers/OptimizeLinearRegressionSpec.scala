package io.scanet.optimizers

import breeze.linalg._
import io.scanet.core.func._
import io.scanet.syntax._
import io.scanet.test.CustomMatchers
import org.scalatest.FlatSpec

import scala.concurrent.ExecutionContext.Implicits._

class OptimizeLinearRegressionSpec extends FlatSpec with CustomMatchers {

  "SGD with full batch" should "find a minimum of [linear regression] within 1500 iterations" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = SGD(coef.rows, 0.01)
      .minimize(linearRegression, coef)
      .through(iter(1500))
      .observe(logStdOut)
      .observe(plotToFile("GD:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 0.01)
  }

  "SGD with one item in batch" should "find a minimum of [linear regression] within 1500 iterations" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = SGD(1, 0.005)
      .minimize(linearRegression, coef)
      .through(iter(1500))
      .observe(logStdOut)
      .observe(plotToFile("SGD:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 50)
  }

  "SGD with 25 items in batch" should "find a minimum of [linear regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = SGD(20, 0.005)
      .minimize(linearRegression, coef)
      .through(epoch(25))
      .observe(logStdOut)
      .observe(plotToFile("MiniBatch:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 3)
  }

  "SGD with momentum" should "find a minimum of [linear regression] within 50 epochs" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = SGD(20, 0.005, 0.2)
      .minimize(linearRegression, coef)
      .through(epoch(25))
      .observe(logStdOut)
      .observe(plotToFile("Momentum:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 3)
  }

  "SGD with Nesterov momentum" should "find a minimum of [linear regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = SGD(20, 0.005, 0.2, nesterov = true)
      .minimize(linearRegression, coef)
      .through(epoch(25))
      .observe(logStdOut)
      .observe(plotToFile("Nesterov:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 3)
  }

  "AdaGrad" should "find a minimum of [linear regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = AdaGrad(20, 1)
      .minimize(linearRegression, coef)
      .through(epoch(25))
      .observe(logStdOut)
      .observe(plotToFile("AdaGrad:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 3)
  }

  "AdaDelta" should "find a minimum of [linear regression] within 250 epochs" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = AdaDelta(20, 1, 0.9)
      .minimize(linearRegression, coef)
      .through(epoch(250))
      .observe(logStdOut)
      .observe(plotToFile("AdaDelta:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 3)
  }

  "RMSProp" should "find a minimum of [linear regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = RMSProp(20, 0.1, 0.9)
      .minimize(linearRegression, coef)
      .through(epoch(25))
      .observe(logStdOut)
      .observe(plotToFile("RMSProp:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 3)
  }

  "Adam" should "find a minimum of [linear regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = Adam(20, 0.1, 0.9, 0.999)
      .minimize(linearRegression, coef)
      .through(epoch(25))
      .observe(logStdOut)
      .observe(plotToFile("Adam:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 3)
  }

  "Adamax" should "find a minimum of [linear regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = Adamax(20, 0.1, 0.9, 0.999)
      .minimize(linearRegression, coef)
      .through(epoch(25))
      .observe(logStdOut)
      .observe(plotToFile("Adamax:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 3)
  }

  "Nadam" should "find a minimum of [linear regression] within 25 epochs" in {
    val read = breeze.linalg.csvread(resource("linear_function_1.scv"))
    val coef = DenseMatrix.horzcat(DenseMatrix.ones[Double](read.rows, 1), read)
    val vars = Nadam(20, 0.1, 0.9, 0.999)
      .minimize(linearRegression, coef)
      .through(epoch(25))
      .observe(logStdOut)
      .observe(plotToFile("Nadam:MSR.png"))
      .runSync.vars
    linearRegression(coef)(vars) should beWithinTolerance(4.48, 3)
  }


}