package io.scanet.optimizers

import breeze.linalg.{DenseMatrix, DenseVector}
import simulacrum.typeclass
import fs2.Stream
import cats.effect.IO
import io.scanet.func.DiffFunction
import io.scanet.func.DiffFunction.DFBuilder

@typeclass
trait Optimizer[A] {

  def minimize[B: DiffFunction](op: A, f: DFBuilder[B], coef: DenseMatrix[Double], initVars: Option[DenseVector[Double]] = None): Stream[IO, Event[B]] =
    optimize(op, f, coef, initVars, min = true)

  def maximize[B: DiffFunction](op: A, f: DFBuilder[B], coef: DenseMatrix[Double], initVars: Option[DenseVector[Double]] = None): Stream[IO, Event[B]] =
    optimize(op, f, coef, initVars, min = false)

  def optimize[B: DiffFunction](op: A, f: DFBuilder[B], coef: DenseMatrix[Double], initVars: Option[DenseVector[Double]] = None, min: Boolean): Stream[IO, Event[B]] = {
    val vars = initVars.getOrElse({
      val arity = DiffFunction[B].arity(f(coef))
      DenseVector.rand[Double](arity)
    })
    optimize(op, f, coef, vars, min)
  }

  def optimize[B: DiffFunction](op: A, f: DFBuilder[B], coef: DenseMatrix[Double], initVars: DenseVector[Double], min: Boolean): Stream[IO, Event[B]]
}

case class Result(epoch: Int, iter: Int, vars: DenseVector[Double])

case class Given[A](f: DFBuilder[A], coef: DenseMatrix[Double])

case class Event[A](result: Result, given: Given[A], other: Any = null) {
  def map(f: Result => Result): Event[A] = copy(result = f(result))
  def mapWithOther(f: Result => (Result, Any)): Event[A] = {
    val (r, o) = f(result)
    copy(result = r).copy(other = o)
  }
}

