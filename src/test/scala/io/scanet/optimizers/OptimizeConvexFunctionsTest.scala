package io.scanet.optimizers

import io.scanet.syntax._
import io.scanet.core.func._
import io.scanet.test.CustomMatchers
import org.scalatest.FlatSpec
import scala.concurrent.ExecutionContext.Implicits._

class OptimizeConvexFunctionsTest extends FlatSpec with CustomMatchers {

  "a gradient decent" should "find a minimum of a function x^2 within 10 iterations" in {
    val vars = SGD(rate = 0.3)
      .minimize(polynomial, `x0^2`)
      .through(iter(10))
      .observe(logStdOut)
      .observe(plotToFile("SGD:x^2.png"))
      .runSync.vars
    polynomial(`x0^2`)(vars) should beWithinTolerance(0.0, 0.001)
  }

  it should "find a minimum of a function x^2 with 0.001 convergence delta" in {
    val Result(n, i, vars) = SGD(rate = 0.3)
      .minimize(polynomial, `x0^2`)
      .through(convergeDelta(0.001))
      .observe(logStdOut)
      .runSync
    polynomial(`x0^2`)(vars) should beWithinTolerance(0.0, 0.001)
  }

  it should "find a minimum of a function (x0^2 + 5x1^2 + 10) within 40 iterations" in {
    val vars = SGD(rate = 0.1)
      .minimize(polynomial, `x0^2 + 5*x1^2 + 10`)
      .through(iter(40))
      .observe(logStdOut)
      .runSync.vars
    polynomial(`x0^2 + 5*x1^2 + 10`)(vars) should beWithinTolerance(10.0, 0.001)
  }
}