package io.scanet.func

trait FunctionsInst
  extends DefaultFunctionsInst
    with LinearFunctionInst
    with PolynomialFunctionInst
    with LinearRegressionFunctionInst
    with LogisticRegressionFunctionInst
    with L1FunctionInst
    with L2FunctionInst
    with SigmoidInst
