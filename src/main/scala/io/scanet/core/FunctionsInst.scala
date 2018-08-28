package io.scanet.core

import io.scanet.core.func._

trait FunctionsInst
  extends OtherFunctionsInst
    with ZeroFunctionInst
    with LinearFunctionInst
    with PolynomialFunctionInst
    with LinearRegressionFunctionInst
    with LogisticRegressionFunctionInst
    with L1FunctionInst
    with L2FunctionInst
    with SigmoidInst
