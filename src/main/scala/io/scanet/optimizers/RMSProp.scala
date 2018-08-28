package io.scanet.optimizers

import breeze.linalg.DenseVector.fill
import breeze.linalg._
import breeze.numerics._
import cats.effect.IO
import fs2.Stream
import io.scanet.core.FunctionsSyntax
import io.scanet.core.DiffFunction
import io.scanet.core.DFBuilder

case class RMSProp(batch: Int = 256, rate: Double = 0.001, rho: Double = 0.9)

trait RMSPropInst extends FunctionsSyntax {

  implicit def RMSPropOptimizer: Optimizer[RMSProp] = new Optimizer[RMSProp] {
    override def optimize[B: DiffFunction](op: RMSProp, f: DFBuilder[B], coef: DenseMatrix[Double],
                                           initVars: DenseVector[Double], min: Boolean): Stream[IO, Event[B]] = {
      def range(from: Int, size: Int): DenseMatrix[Double] = {
        if (coef.rows == 0)
          DenseMatrix.zeros(0, 0)
        else
          coef(from until math.min(from + size, coef.rows), ::)
      }
      val sign = if (min) -1.0 else 1.0
      Stream.iterateEval[IO, Event[B]](Event(Result(0, 0, initVars), Given(f, coef), fill(initVars.length, 0.0)))(prev => {
        IO {
          val Result(epoch, i, vars) = prev.result
          val gradient: DenseVector[Double] =f(range(i, op.batch)).gradient(vars)
          prev.mapWithOther(_ => {
            val gradientAcc = op.rho * prev.other.asInstanceOf[DenseVector[Double]] + (1 - op.rho) * pow(gradient, 2)
            val rates = op.rate / (pow(gradientAcc, 0.5) + 1e-7)
            val updatedArgs = vars + sign * (rates *:* gradient)
            if (coef.rows <= i + op.batch)
              (Result(epoch + 1, 0, updatedArgs), gradientAcc)
            else
              (Result(epoch, i + op.batch, updatedArgs), gradientAcc)
          })
        }
      })
    }
  }
}