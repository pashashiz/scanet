package io.scanet.optimizers

import breeze.linalg._
import io.scanet.func._
import io.scanet.syntax._
import io.scanet.test.CustomMatchers
import org.scalatest.FlatSpec

import scala.concurrent.ExecutionContext.Implicits._

class OptimizeSaddlePointFunctionsTest extends FlatSpec with CustomMatchers {

  "SGD" should "stack in a saddle point of x^4 - 2x^2 + y^2 function" in {
    val vars = SGD()
      .minimize(polynomial, `x0^4 - 2*x0^2 + x1^2`, Some(DenseVector(0.0001, 0.0001)))
      .through(iter(100))
      .observe(logStdOut)
      .observe(plotToFile("SGD:x^4 - 2x^2 + y^2.png"))
      .runSync.vars
    polynomial(`x0^4 - 2*x0^2 + x1^2`)(vars) should beWithinTolerance(0.0, 0.1)
  }

  "SGD with momentum" should "pass a saddle point of x^4 - 2x^2 + y^2 function" in {
    val vars = SGD(rate = 0.1, momentum = 0.9)
      .minimize(polynomial, `x0^4 - 2*x0^2 + x1^2`, Some(DenseVector(0.0001, 0.0001)))
      .through(iter(100))
      .observe(logStdOut)
      .observe(plotToFile("Momentum:x^4 - 2x^2 + y^2.png"))
      .runSync.vars
    polynomial(`x0^4 - 2*x0^2 + x1^2`)(vars) should beWithinTolerance(-1.0, 0.1)
  }

  "SGD with Nesterov momentum" should "pass a saddle point of x^4 - 2x^2 + y^2 function" in {
    val vars = SGD(momentum = 0.9, nesterov = true)
      .minimize(polynomial, `x0^4 - 2*x0^2 + x1^2`, Some(DenseVector(0.0001, 0.0001)))
      .through(iter(100))
      .observe(logStdOut)
      .observe(plotToFile("Nesterov:x^4 - 2x^2 + y^2.png"))
      .runSync.vars
    polynomial(`x0^4 - 2*x0^2 + x1^2`)(vars) should beWithinTolerance(-1.0, 0.1)
  }

  "RMSProp" should "pass a saddle point of x^4 - 2x^2 + y^2 function" in {
    val vars = RMSProp(rate = 0.1)
      .minimize(polynomial, `x0^4 - 2*x0^2 + x1^2`, Some(DenseVector(0.0001, 0.0001)))
      .through(iter(100))
      .observe(logStdOut)
      .observe(plotToFile("RMSProp:x^4 - 2x^2 + y^2.png"))
      .runSync.vars
    polynomial(`x0^4 - 2*x0^2 + x1^2`)(vars) should beWithinTolerance(-1.0, 0.1)
  }

  "AdaGrad" should "pass a saddle point of x^4 - 2x^2 + y^2 function" in {
    val vars = AdaGrad(rate = 1)
      .minimize(polynomial, `x0^4 - 2*x0^2 + x1^2`, Some(DenseVector(0.0001, 0.0001)))
      .through(iter(100))
      .observe(logStdOut)
      .observe(plotToFile("AdaGrad:x^4 - 2x^2 + y^2.png"))
      .runSync.vars
    polynomial(`x0^4 - 2*x0^2 + x1^2`)(vars) should beWithinTolerance(-1.0, 0.1)
  }

  "AdaDelta" should "pass a saddle point of x^4 - 2x^2 + y^2 function" in {
    val vars = AdaDelta(rho = 0.9999)
      .minimize(polynomial, `x0^4 - 2*x0^2 + x1^2`, Some(DenseVector(0.0001, 0.0001)))
      .through(iter(1000))
      .observe(logStdOut)
      .observe(plotToFile("AdaDelta:x^4 - 2x^2 + y^2.png"))
      .runSync.vars
    polynomial(`x0^4 - 2*x0^2 + x1^2`)(vars) should beWithinTolerance(-1.0, 0.1)
  }

  "Adam" should "pass a saddle point of x^4 - 2x^2 + y^2 function" in {
    val vars = Adam(rate = 0.01, beta2 = 0.9)
      .minimize(polynomial, `x0^4 - 2*x0^2 + x1^2`, Some(DenseVector(0.0001, 0.0001)))
      .through(iter(100))
      .observe(logStdOut)
      .observe(plotToFile("Adam:x^4 - 2x^2 + y^2.png"))
      .runSync.vars
    polynomial(`x0^4 - 2*x0^2 + x1^2`)(vars) should beWithinTolerance(-1.0, 0.1)
  }

  "Adamax" should "pass a saddle point of x^4 - 2x^2 + y^2 function" in {
    val vars = Adamax(rate = 0.01, beta2 = 0.9)
      .minimize(polynomial, `x0^4 - 2*x0^2 + x1^2`, Some(DenseVector(0.0001, 0.0001)))
      .through(iter(100))
      .observe(logStdOut)
      .observe(plotToFile("Adamax:x^4 - 2x^2 + y^2.png"))
      .runSync.vars
    polynomial(`x0^4 - 2*x0^2 + x1^2`)(vars) should beWithinTolerance(-1.0, 0.1)
  }

}