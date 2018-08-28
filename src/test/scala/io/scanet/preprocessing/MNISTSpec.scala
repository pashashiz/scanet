package io.scanet.preprocessing

import breeze.linalg.{*, DenseMatrix}
import org.scalatest.FlatSpec

class MNISTSpec extends FlatSpec {

  "MNIST training set" should "be downloaded" in {
    val set = MNIST.loadTrainingSet(20)
    set.input(*, ::).foreach(v => printImage(v.toDenseMatrix.reshape(28, 28).t))
    set.labels(*, ::).foreach(v => println(v))
  }

  "MNIST test set" should "be downloaded" in {
    val set = MNIST.loadTestSet(20)
    set.input(*, ::).foreach(v => printImage(v.toDenseMatrix.reshape(28, 28).t))
    set.labels(*, ::).foreach(v => println(v))
  }

  def printImage(m: DenseMatrix[Double]): Unit = {
    m(*, ::).foreach(row => {
      row.foreach(el => {
        if (el == 0.0)
          print(" ")
        else
          print("x")
      })
      println()
    })
  }

}
