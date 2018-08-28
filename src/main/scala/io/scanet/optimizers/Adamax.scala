package io.scanet.optimizers

import breeze.linalg.DenseVector.fill
import breeze.linalg._
import breeze.numerics._
import cats.effect.IO
import fs2.Stream
import io.scanet.core.DFBuilder
import io.scanet.core.{DiffFunction, FunctionsSyntax}


case class Adamax(batch: Int = 256, rate: Double = 0.001, beta1: Double = 0.9, beta2: Double = 0.999)

trait AdamaxInst extends FunctionsSyntax {

  implicit def AdamaxOptimizer: Optimizer[Adamax] = new Optimizer[Adamax] {
    override def optimize[B: DiffFunction](op: Adamax, f: DFBuilder[B], coef: DenseMatrix[Double],
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
            val moment2Max = max(op.beta2 * prevMoment2, (1 - op.beta2) * abs(gradient))
            val updatedArgs = vars + sign * op.rate * moment1Unbiased /:/ (moment2Max + 1e-7)
            if (coef.rows <= i + op.batch)
              (Result(epoch + 1, 0, updatedArgs), (moment1, moment2Max))
            else
              (Result(epoch, i + op.batch, updatedArgs), (moment1, moment2Max))
          })
        }
      })
    }
  }
}