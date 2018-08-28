name := "scala-net"
version := "0.1"
scalaVersion := "2.12.4"

scalacOptions += "-Ypartial-unification"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies ++= Seq(
  "com.github.mpilquist" %% "simulacrum" % "0.11.0",
  "org.typelevel" %% "cats-core" % "1.0.1",
  "co.fs2" %% "fs2-core" % "0.10.1",
  //  "org.typelevel" %% "spire" % "0.14.1",
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2",
//  "com.sksamuel.scrimage" %% "scrimage-core" % "2.1.8",
  "org.scalacheck" %% "scalacheck" % "1.13.5" % "test",
  "org.scalatest" %% "scalatest" % "3.0.5" % "test"
)
