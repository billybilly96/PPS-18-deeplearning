package it.unibo.pps18.deeplearning.entities

import scala.util.{Success, Try}

sealed trait NeuralNetwork {
  def params: Int
  def isEmpty: Boolean
}
object NeuralNetwork {
  /** A representation of a neural network.
   *
   *  @constructor create a neural network by specifying its input `i`, hidden `h`, and output `o` neurons
   *  @param i the number of input neurons
   *  @param h the number of hidden neurons and layers
   *  @param o the number of output neurons
   */
  case class MyNeuralNetwork(i: Int, h: List[Int], o: Int) extends NeuralNetwork {
    require(!isEmpty)

    /** Describe a neural network.
     *
     *  @return the description.
     */
    override def toString: String = s"${getClass.getSimpleName}: $i => $h => $o, with a total of $params parameters"

    /** Compute the number of parameters of the neural network.
     *
     *  @return the number of parameters.
     */
    override def params: Int = h.isEmpty match {
      case true => i * o + o
      case _ => i * h.head + getMulFromList(h) + h.last * o
    }

    /** Check if the neural network is empty, hence
     *
     *  @return if the neural network is empty.
     */
    override def isEmpty: Boolean = i == 0 || o == 0

    /** Compute the multiplication of values inside a list sequentially.
     *
     *  @param l the list to perform the computation
     *  @return the multiplication of values.
     */
    private def getMulFromList(l: List[Int]): Int = l.size match {
      case x if x > 1 => l.zip(l.tail).map(x => x._1 * x._2).reduceLeft(_ + _)
      case _ => l.head
    }
  }

  /** Check if a neural network is valid.
   *
   *  @param o an option that maybe contains a neural network
   *  @return if a neural network is valid.
   */
  def isValid(o: Option[NeuralNetwork]): Boolean = o match {
    case Some(d) => !d.isEmpty
    case _ => false
  }

  /** Implicit class in order to build a custom interpolator for creating a neural network.
   *
   *  @constructor create an interpolator
   *  @param sc the string context to create the interpolator
   */
  implicit class NeuralNetworkInterpolator(sc: StringContext) {
    /** Build an interpolator `nn` for creating a neural network.
     *
     *  @param args any arguments to process
     *  @return an option of a neural network if the processing is successful.
     */
    def nn(args: Any*): Option[MyNeuralNetwork] = {
      Try {
        val totalString: String = sc.s(args: _*)
        val tokens: Array[String] = totalString.split(";")
        Some(MyNeuralNetwork(
          tokens(0).toInt,
          if (tokens(1).isEmpty) List() else tokens(1).split(",").map(x => x.toInt).filter(_ != 0).toList,
          tokens(2).toInt
        ))
      } match {
        case Success(x) => x
        case _ => None
      }
    }
  }
}


