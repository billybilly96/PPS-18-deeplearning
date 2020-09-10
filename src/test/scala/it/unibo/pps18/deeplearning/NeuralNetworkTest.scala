package it.unibo.pps18.deeplearning

import it.unibo.pps18.deeplearning.entities.NeuralNetwork._

import org.scalatest.FunSpec

/** Test class to check the correctness of neural network representation. */
class NeuralNetworkTest extends FunSpec {

  val i: String = "10"
  val h: String = "1,2,3"
  val o: String = "1"

  val nn = nn"$i;$h;$o"
  val wrongNn = nn"${0};$h;$o"

  describe("A neural network") {
    describe("when correct") {
      it("should be an instance of MyNeuralNetwork") {
        assert(nn.getOrElse().isInstanceOf[MyNeuralNetwork])
      }
      it("should have a depth of (i×h + h×o) => (10*1 + 1*2 + 2*3 + 3*1) = 21") {
        assert(nn.get.params == 21)
      }
    }
    describe("when not correct") {
      it("should not be an instance of MyNeuralNetwork") {
        assert(!wrongNn.getOrElse().isInstanceOf[MyNeuralNetwork])
      }
    }
  }

}
