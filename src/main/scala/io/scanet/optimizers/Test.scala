package io.scanet.optimizers

import breeze.linalg.DenseMatrix
import simulacrum.typeclass

case class User[A](name: String, metadata: Option[A] = /*_*/ Some("data") /*_*/)

@typeclass
trait Show[A] {
  def show(a: A): String
}

object ShowInst {
  implicit def showUser[A]: Show[User[A]] =
    (a: User[A]) => a.name + ": " + a.metadata
}

object Test {
  def main(args: Array[String]): Unit = {
    import io.scanet.optimizers.Show.ops._
    import io.scanet.optimizers.ShowInst.showUser
    println(User("joe").show)

    val m = DenseMatrix((1, 2, 3), (4, 5, 6))
    println(m(::, 0).toDenseMatrix.t)

    // DataSet(input, labels)
    // MNIST(training: DataSet, test: DataSet)
    //
    // preprocessors.loadMNIST(percent)
    // preprocessors.loadMNIST(trainingSize, testSize)

  }
}