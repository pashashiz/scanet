package io.scanet.nn

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.pow
import io.scanet.core.DiffFunction

case class NNError[A: Layer](layer: A, in: DenseMatrix[Double], out: DenseMatrix[Double])

trait NNErrorFunctionInst extends Layer.ToLayerOps {

  implicit def NNErrorFunctionInst[A: Layer]: DiffFunction[NNError[A]] = new DiffFunction[NNError[A]] {

    override def gradient(f: NNError[A], vars: DenseVector[Double]): DenseVector[Double] = {
      var theta = f.layer.unpack(f.in.cols)(vars)
      var error =  f.layer.forward(theta, f.in) - f.out
      var (_, grad) = f.layer.backward(theta, f.in, error)
      val penalty = f.layer.backwardPenalty(theta)
      f.layer.pack(grad) + f.layer.pack(penalty)
    }

    override def apply(f: NNError[A], vars: DenseVector[Double]): Double = {
      var theta = f.layer.unpack(f.in.cols)(vars)
      var realOut = f.layer.forward(theta, f.in)
      val error = realOut - f.out
      val errorPerLabel = error(::, *).map(column => {
        val squaredErrors: DenseVector[Double] = pow(column, 2)
        sum(squaredErrors) / (2 * column.length)
      })
      val penalty = f.layer.forwardPenalty(theta)
      sum(errorPerLabel) + penalty
    }

    override def arity(f: NNError[A]): Int = {
      def foldLeft(acc: Int, shape: List[Shape]): Int = shape match {
        case Nil => acc
        case first::second::tail =>
          val firstUnits = if (second.bias) first.units + 1 else first.units
          foldLeft(acc + firstUnits * second.units, if (tail != Nil) second::tail else Nil)
      }
      val fullShape = Shape(f.in.cols, bias = false) :: f.layer.shape
      foldLeft(0, fullShape)
    }
  }
}
