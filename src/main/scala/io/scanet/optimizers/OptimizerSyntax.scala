package io.scanet.optimizers

import java.io.PrintStream

import breeze.linalg.DenseVector
import cats.effect.{IO, Sync}
import fs2.{Pipe, Pull, Segment, Sink, Stream}
import io.scanet.core.DiffFunction
import io.scanet.core.DiffFunction.ops._

import scala.language.{higherKinds, implicitConversions}

trait OptimizerSyntax extends Optimizer.ToOptimizerOps {

  class EventStreamOpt[A](stream: Stream[IO, Event[A]]) {
    def runSync: Result = stream.compile.last.unsafeRunSync().get.result
    def runAndGetVars: DenseVector[Double] = stream.compile.last.unsafeRunSync().get.result.vars
  }

  implicit def toEventStreamOpt[A](stream: Stream[IO, Event[A]]): EventStreamOpt[A] = new EventStreamOpt(stream)

  def logStdOut[F[_], A: DiffFunction](implicit F: Sync[F]): Sink[F, Event[A]] = log(System.out)

  def log[F[_], A: DiffFunction](out: PrintStream)(implicit F: Sync[F]): Sink[F, Event[A]] =
    in => {
      in.covary[F].flatMap(event => Stream.eval(F.delay({
        val Event(Result(epoch, i, vars), Given(f, coef), _) = event
        val gradient = f(coef).gradient(vars)
        val value = f(coef)(vars)
        out.print(s"$epoch:$i\t f = $value\n")
      })))
    }

  def plotToFile[F[_], A: DiffFunction](dest: String)(implicit F: Sync[F]): Sink[F, Event[A]] =
    in => {
      in.covary[F].fold((List[Double](), List[Double]()))((acc, event) => {
        val (x, y) = acc
        val n =  x.length
        val Event(Result(epoch, i, vars), Given(f, coef), _) = event
        val gradient = f(coef).gradient(vars)
        val value = f(coef)(vars)
        (x.headOption.getOrElse(0.0) + 1.0 :: x, value :: y)
      }).flatMap(acc => Stream.eval(F.delay({
        import breeze.plot._
        val figure = Figure()
        figure.width = 1920
        figure.height = 1080
        val p = figure.subplot(0)
        p.xlabel = "n"
        p.ylabel = "f()"
        val (x, y) = acc
        p += plot(x, y, colorcode = "m")
        figure.saveas(dest, 96)
      })))
    }

  def takeWhile[F[_], A, B](acc: B, mappend: (B, Event[A]) => B)
                           (enough: B => Boolean): Pipe[F, Event[A], Event[A]] = {
    def go(s: Stream[F, Event[A]], acc: B): Pull[F, Event[A], Unit] = {
      /*_*/
      s.pull.uncons1.flatMap {
        case Some((event, tail)) =>
          if (enough(acc))
            Pull.done
          else
            Pull.segment(Segment(event))
              .flatMap(_ => go(tail, mappend(acc, event)))
        case None => Pull.done
      }
      /*_*/
    }
    in => go(in, acc).stream
  }

  def epoch[F[_], A](n: Int): Pipe[F, Event[A], Event[A]] =
    takeWhile[F, A, Int](0, (_: Int, event) => event.result.epoch)(_ >= n)

  def iter[F[_], A](n: Int): Pipe[F, Event[A], Event[A]] =
    takeWhile[F, A, Int](0, (acc: Int, event) => acc + 1)(_ >= n - 1)

  case class EventHistory[A](events: List[Event[A]])

  def convergeDelta[F[_], A: DiffFunction](error: Double): Pipe[F, Event[A], Event[A]]  = {
    takeWhile[F, A, List[Event[A]]](List(), (acc: List[Event[A]], event) => (event :: acc).take(2))({
      case current::prev::tl =>
        val Event(_, Given(f, coef), _) = current
        math.abs(f(coef)(current.result.vars) - f(coef)(prev.result.vars)) < error
      case _ => false
    })
  }
}
