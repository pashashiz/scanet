package io.scanet.preprocessing

import java.io.{DataInputStream, FileInputStream}
import java.net.URL
import java.nio.channels.{Channels, FileChannel}
import java.nio.file.StandardOpenOption.WRITE
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream

import breeze.linalg.{DenseMatrix, DenseVector}

case class MNIST(training: DataSet, test: DataSet)

object MNIST {

  def load(trainingSize: Int, testSize: Int): MNIST =
    MNIST(loadTrainingSet(trainingSize), loadTestSet(testSize))

  def loadTrainingSet(size: Int): DataSet = {
    DataSet(
      loadImages("train-images-idx3-ubyte.gz", size),
      loadLabels("train-labels-idx1-ubyte.gz", size))
  }

  def loadTestSet(size: Int): DataSet = {
    DataSet(
      loadImages("t10k-images-idx3-ubyte.gz", size),
      loadLabels("t10k-labels-idx1-ubyte.gz", size))
  }

  def loadImages(name: String, size: Int): DenseMatrix[Double] = {
    openStream(downloadOrCached(name), stream => {
      require(stream.readInt() == 2051, "wrong MNIST image stream magic number")
      val count = stream.readInt()
      val width = stream.readInt()
      val height = stream.readInt()
      require(size <= count, "passed size is bigger than the actual data set")
      val images = DenseMatrix.zeros[Double](size, width * height)
      for (x <- 0 until size) {
        images(x, ::) := readImage(stream, height * width).t
      }
      images
    })
  }

  def loadLabels(name: String, size: Int): DenseMatrix[Double] = {
    openStream(downloadOrCached(name), stream => {
      require(stream.readInt() == 2049, "wrong MNIST image stream magic number")
      val count = stream.readInt()
      require(size <= count, "passed size is bigger than the actual data set")
      val labels = DenseMatrix.zeros[Double](size, 10)
      for (x <- 0 until size) {
        labels(x, ::) := readLabel(stream, 10).t
      }
      labels
    })
  }

  private def openStream[A](path: Path, f: DataInputStream => A) = {
    val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path.toString)))
    try {
      f.apply(stream)
    } finally {
      stream.close()
    }
  }

  def readImage(stream: DataInputStream, size: Int): DenseVector[Double] = {
    val m = DenseVector.zeros[Double](size)
    for (x <- 0 until size) {
      m(x) = stream.readUnsignedByte().toDouble
    }
    m
  }

  def readLabel(stream: DataInputStream, size: Int): DenseVector[Double] = {
    val m = DenseVector.zeros[Double](size)
    val value = stream.readUnsignedByte()
    m(value) = 1
    m
  }

  private def downloadOrCached(name: String): Path = {
    val resource = Channels.newChannel(new URL(s"http://yann.lecun.com/exdb/mnist/$name").openStream())
    val dir = Paths.get(System.getProperty("user.home"), "mnist")
    Files.createDirectories(dir)
    val file = dir.resolve(name)
    if (!Files.exists(file)) {
      Files.createFile(file)
      FileChannel.open(file, WRITE).transferFrom(resource, 0, Long.MaxValue)
      file
    } else {
      file
    }
  }

}