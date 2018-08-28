package io.scanet.optimizers

import breeze.linalg.DenseVector.fill
import breeze.linalg.{DenseMatrix, DenseVector}
import cats.effect.IO
import fs2.Stream
import io.scanet.core.FunctionsSyntax
import io.scanet.core.DiffFunction
import io.scanet.core.DFBuilder

case class SGD(batch: Int = 256, rate: Double = 0.01, momentum: Double = 0.0, nesterov: Boolean = false)

trait SGDInst extends FunctionsSyntax {

  implicit def SGDOptimizer: Optimizer[SGD] = new Optimizer[SGD] {
    override def optimize[B: DiffFunction](op: SGD, f: DFBuilder[B], coef: DenseMatrix[Double],
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
          val gradient = f(range(i, op.batch)).gradient(vars)
          prev.mapWithOther(_ => {
            val velocityPrev = prev.other.asInstanceOf[DenseVector[Double]]
            val velocity = op.momentum * velocityPrev + op.rate * gradient
            val updatedArgs = if (op.nesterov)
              vars + sign * (op.momentum * velocityPrev + (1 + op.momentum) * velocity)
            else
              vars + sign * velocity
            if (coef.rows <= i + op.batch)
              (Result(epoch + 1, 0, updatedArgs), velocity)
            else
              (Result(epoch, i + op.batch, updatedArgs), velocity)
          })
        }
      })
    }
  }
}
