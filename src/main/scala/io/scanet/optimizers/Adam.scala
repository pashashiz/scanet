package io.scanet.optimizers

import breeze.linalg.DenseVector.fill
import breeze.linalg._
import breeze.numerics._
import cats.effect.IO
import fs2.Stream
import io.scanet.core.{DiffFunction, FunctionsSyntax}
import io.scanet.core.DFBuilder


case class Adam(batch: Int = 256, rate: Double = 0.001, beta1: Double = 0.9, beta2: Double = 0.999)

trait AdamInst extends FunctionsSyntax {

  implicit def AdamOptimizer: Optimizer[Adam] = new Optimizer[Adam] {
    override def optimize[B: DiffFunction](op: Adam, f: DFBuilder[B], coef: DenseMatrix[Double],
                                           initVars: DenseVector[Double], min: Boolean): Stream[IO, Event[B]] = {
      def range(from: Int, size: Int): DenseMatrix[Double] = {
        if (coef.rows == 0)
          DenseMatrix.zeros(0, 0)
        else
          coef(from until math.min(from + size, coef.rows), ::)
      }
      val sign = if (min) -1.0 else 1.0
      Stream.iterateEval[IO, Event[B]](Event(Result(0, 0, initVars), Given(f, coef),
        (fill(initVars.length, 0.0), fill(initVars.length, 0.0))))(prev => {
        IO {
          val Result(epoch, i, vars) = prev.result
          val gradient: DenseVector[Double] = f(range(i, op.batch)).gradient(vars)
          prev.mapWithOther(_ => {
            val iter = (epoch * coef.rows + i) / op.batch + 1
            val (prevMoment1, prevMoment2) = prev.other.asInstanceOf[(DenseVector[Double], DenseVector[Double])]
            val moment1 = op.beta1 * prevMoment1 + (1 - op.beta1) * gradient
            val moment1Unbiased = moment1 / (1.0 - pow(op.beta1, iter))
            val moment2 = op.beta2 * prevMoment2 + (1 - op.beta2) * pow(gradient, 2)
            val moment2Unbiased = moment2 / (1.0 - pow(op.beta2, iter))
            val updatedArgs = vars + sign * op.rate * moment1Unbiased /:/ (pow(moment2Unbiased, 0.5) + 1e-7)
            if (coef.rows <= i + op.batch)
              (Result(epoch + 1, 0, updatedArgs), (moment1, moment2))
            else
              (Result(epoch, i + op.batch, updatedArgs), (moment1, moment2))
          })
        }
      })
    }
  }
}