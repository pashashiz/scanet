package io.scanet.preprocessing

import breeze.linalg.DenseMatrix

case class DataSet(input: DenseMatrix[Double], labels: DenseMatrix[Double])