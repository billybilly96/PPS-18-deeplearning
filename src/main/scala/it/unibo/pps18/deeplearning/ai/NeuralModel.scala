package it.unibo.pps18.deeplearning.ai

import it.unibo.pps18.deeplearning.entities.Dataset._
import it.unibo.pps18.deeplearning.entities.NeuralNetwork._

import scala.collection.mutable.ListBuffer
import scala.concurrent.Await
import scala.concurrent.duration.Duration
import scala.concurrent.ExecutionContext.Implicits.global

import scalaz.std.stream._

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits.DoubleArrayMtrix2INDArray

import com.thoughtworks.feature.Factory
import com.thoughtworks.deeplearning.plugins.Builtins
import com.thoughtworks.deeplearning.plugins.INDArrayWeights
import com.thoughtworks.each.Monadic._
import com.thoughtworks.future._

/** Object to perform forecasting of a time series using NeuralModel.scala library. */
object NeuralModel {

  /** Forecast future values of the time series after having trained the neural model.
   *
   *  @param dataset the dataset to use for forecasting
   *  @param elementsToPredict the number of future values to forecast
   *  @return a tuple composed of the neural model predictions on the test set and the forecasted values.
   */
  def getForecasting(dataset: TimeSeriesDataset, elementsToPredict: Int): (List[Double], List[Double]) = {

    val (xTrain, yTrain) = dataset.trainSet
    val (xTest, _) = dataset.testSet

    val inputNeurons: Int = xTrain.getRow(0).length
    val hiddenNeurons: Int = 0
    val outputNeurons: Int = 1

    // Useful for the neural network representation.
    val nn = nn"$inputNeurons;$hiddenNeurons;$outputNeurons"
    if (isValid(nn)) println(nn.get.toString) else println("The neural network is not correct")

    // Fixed learning rate plugin.
    import $exec.`https://gist.github.com/Atry/1fb0608c655e3233e68b27ba99515f16/raw/39ba06ee597839d618f2fcfe9526744c60f2f70a/FixedLearningRate.sc`
    // Adam optimizer plugin.
    import $exec.`https://gist.github.com/Rabenda/0c2fc6ba4cfa536e4788112a94200b50/raw/233cbc83932dad659519c80717d145a3983f57e1/Adam.sc`

    // Global configuration for the neural network.
    val hyperparameters = Factory[Builtins with FixedLearningRate with Adam].newInstance(learningRate = 0.01)

    // Import plugin's implicits
    import hyperparameters._
    import hyperparameters.implicits._

    // Weights of the neural network.
    val weights = INDArrayWeight(Nd4j.randn(inputNeurons, outputNeurons, 1))

    /** Build the ReLU activation function.
     *
     *  @param input the input to rectify
     *  @return the max between 0 and the input.
     */
    def relu(input: INDArrayLayer): INDArrayLayer = max(0.0, input)

    /** Compose a neural network.
     *
     *  @param inputs the input data
     *  @return a layer of the neural network, seen as the dot product between inputs and weights.
     */
    def neuralNetwork(inputs: INDArray, weights: INDArrayWeight): INDArrayLayer = relu(inputs dot weights)

    /** Loss function to determine the mean squared error (MSE) between predictions and the real values.
     *
     *  @param predictions the predictions of the neural network
     *  @param target the real values
     *  @return the MSE.
     */
    def lossFunctionMse(prediction: INDArrayLayer, target: INDArray): DoubleLayer = {
      val error: INDArrayLayer = prediction - target
      (error * error).mean
    }

    /** Build a linear regression model.
     *
     *  @param inputs the input data
     *  @param target the target data
     *  @return a layer of the neural network.
     */
    def linearRegression(inputs: INDArray, target: INDArray): DoubleLayer = {
      val prediction: INDArrayLayer = neuralNetwork(inputs, weights)
      lossFunctionMse(prediction, target)
    }

    val epochs: Int = 5000

    /** Training task. You need to use @monadic[T] annotation in order to enable the for comprehension syntax.
     *
     *  @return a scala future with a stream of the loss values.
     */
    @monadic[Future]
    def trainTask: Future[Stream[Double]] = {
      for (_ <- (0 until epochs).toStream) yield {
        linearRegression(xTrain, yTrain).train.each
      }
    }

    // Training the neural network.
    Await.result(trainTask.toScalaFuture, Duration.Inf)

    // Get the predictions on the test set.
    val predictions: Any = Await.result(neuralNetwork(xTest, weights).predict.toScalaFuture, Duration.Inf)
    val predictionsList: List[Double] = getListOfDoubleFromNumericAny(predictions)

    // Initialize the forecasting list.
    val forecasting = initForecastingList(dataset, xTest)

    /** Forecasting the future values of the time series.
     *
     *  @param data the data to be used for the forecasting
     *  @return a list of the forecasted values.
     */
    def forecast(data: List[List[Double]]): List[Double] = {
      val dataINDArray = data.map(x => x.toArray).toArray.toNDArray
      val res = Await.result(neuralNetwork(dataINDArray, weights).predict.toScalaFuture, Duration.Inf)
      getListOfDoubleFromNumericAny(res)
    }

    val forecastByTime = {
      for (_ <- 0 until elementsToPredict) yield {
        val forecastValues: List[Double] = forecast(forecasting.toList)
        forecasting += forecastValues.takeRight(dataset.nPast)
        forecastValues
      }
    }

    // Denormalize values.
    val denormalizedPredictions: List[Double] = dataset.denormalize(predictionsList)
    val denormalizedForecasting: List[Double] = dataset.denormalize(forecastByTime.toList.last).takeRight(elementsToPredict + 1)

    (denormalizedPredictions, denormalizedForecasting)
  }

  /** Initialize a list in order to forecast future values.
   *
   *  @param dataset the set of data
   *  @param xTest the xTest
   *  @return a list of values.
   */
  private def initForecastingList(dataset: TimeSeriesDataset, xTest: INDArray) = {
    val forecastingList = ListBuffer[List[Double]]()
    xTest.data().asDouble().sliding(dataset.nPast, dataset.nPast).foreach(x => forecastingList += x.toList)
    forecastingList += dataset.normalize(dataset.data.takeRight(dataset.nPast)).toList
    forecastingList
  }

  /** Get a list of double values from data of type Any which contains numeric values.
   *
   *  @param data the data from which to get values
   *  @return a list of double values.
   */
  private def getListOfDoubleFromNumericAny(data: Any): List[Double] = {
    data.toString.substring(1, data.toString.length - 1).split(",").toList.flatMap {
      case x => x.toDouble :: Nil
    }
  }

}
