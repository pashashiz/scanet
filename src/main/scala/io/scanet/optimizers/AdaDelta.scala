package io.scanet.optimizers

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.linalg.DenseVector.fill
import breeze.numerics.pow
import cats.effect.IO
import fs2.Stream
import io.scanet.func.DiffFunction
import io.scanet.func.DiffFunction.DFBuilder
import DiffFunction.ops._

case class AdaDelta(batch: Int = 256, rate: Double = 1, rho: Double = 0.9)

trait AdaDeltaInst {

  implicit def AdaDeltaOptimizer: Optimizer[AdaDelta] = new Optimizer[AdaDelta] {
    override def optimize[B: DiffFunction](op: AdaDelta, f: DFBuilder[B], coef: DenseMatrix[Double],
                                           initVars: DenseVector[Double], min: Boolean): Stream[IO, Event[B]] = {
        def range(from: Int, size: Int): DenseMatrix[Double] = {
          if (coef.rows == 0)
            DenseMatrix.zeros(0, 0)
          else
            coef(from until math.min(from + size, coef.rows), ::)
        }
        val sign = if (min) -1.0 else 1.0
        val start: Event[B] = Event(Result(0, 0, initVars), Given(f, coef), (fill(initVars.length, 0.0), fill(initVars.length, 0.0)))
        Stream.iterateEval[IO, Event[B]](start)(prev => {
          IO {
            val Result(epoch, i, vars) = prev.result
            val gradient: DenseVector[Double] = f(range(i, op.batch)).gradient(vars)
            prev.mapWithOther(_ => {
              val (prevGradientAcc, prevArgAcc) = prev.other.asInstanceOf[(DenseVector[Double], DenseVector[Double])]
              val gradientAcc = op.rho * prevGradientAcc + (1 - op.rho) * pow(gradient, 2)
              val RMSGrad =  pow(gradientAcc + 1e-7, 0.5)
              val prevRMSArg =  pow(prevArgAcc + 1e-7, 0.5)
              val argDelta = op.rate * ((prevRMSArg /:/ RMSGrad) *:* gradient)
              val updatedArgs = vars + sign * argDelta
              val argAcc = op.rho * prevArgAcc + (1 - op.rho) * pow(argDelta, 2)
              if (coef.rows <= i + op.batch)
                (Result(epoch + 1, 0, updatedArgs), (gradientAcc, argAcc))
              else
                (Result(epoch, i + op.batch, updatedArgs), (gradientAcc, argAcc))
            })
          }
        })

    }
  }
}
