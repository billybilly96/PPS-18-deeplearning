package it.unibo.pps18.deeplearning.entities

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits.DoubleArrayMtrix2INDArray

sealed trait Dataset
object Dataset {

  /** A dataset for handle time series analysis.
   *
   *  @constructor create a new dataset for time series by specifying its `data`, `nPast`, and `trainTestSplit`
   *  @param data the dataset values
   *  @param nPast the window width to perform sliding window
   *  @param trainTestSplit the split percentage to create the training set and the test set
   */
  case class TimeSeriesDataset(data: List[Double], nPast:Int, trainTestSplit: Double) extends Dataset {

    val (trainSet, testSet) = getTrainTest(data, nPast, trainTestSplit)

    /** Describe a dataset.
     *
     *  @return the description.
     */
    override def toString: String = s"${getClass.getSimpleName}(min: $min, max: $max): \n$data \n\nTraining set: \n$trainSet \n\nTest set: \n$testSet"

    /** Check if dataset is empty.
     *
     *  @return if dataset is empty.
     */
    def isEmpty: Boolean = data.isEmpty

    /** Find the min value inside the dataset.
     *
     *  @return the min value.
     */
    def min: Double = data.min

    /** Find the max value inside the dataset.
     *
     *  @return the max value.
     */
    def max: Double = data.max

    /** Perform a MinMax normalization on data.
     *
     *  @param data the data to normalize
     *  @return the data normalized.
     */
    def normalize(data: List[Double]): List[Double] = {
      data.map(x => (x - min) / (max - min))
    }

    /** Perform a MinMax denormalization on data to get the original values.
     *
     *  @param data the data to denormalize
     *  @return the data denormalized.
     */
    def denormalize(data: List[Double]): List[Double] = {
      data.map(x => x * (max - min) + min)
    }

    /** Get the training set and the test set from dataset data.
     *
     *  @param data the data to create the training set and the test set
     *  @param nPast the window width to perform sliding window
     *  @param split the split percentage
     *  @return a tuple composed of the training set and the test set.
     */
    private def getTrainTest(data: List[Double], nPast: Int, split: Double): ((INDArray, INDArray), (INDArray, INDArray)) = {
      val normalizedData: List[Double] = normalize(data)
      val cutPoint: Int = (normalizedData.length * split).toInt
      val (trainSplit, testSplit) = normalizedData.splitAt(cutPoint)
      (computeWindows(trainSplit, nPast), computeWindows(testSplit, nPast))
    }

    /** Compute a sliding window matrix from a series of values.
     *
     *  @param data the data to perform windowing
     *  @param nPast the window width
     *  @return a tuple composed of the window matrix to train with the respective value.
     */
    private def computeWindows(data: List[Double], nPast: Int): (INDArray, INDArray) = {
      val x: INDArray = data.sliding(nPast).take(data.length - nPast).map(x => x.toArray).toArray.toNDArray
      val y: INDArray = data.takeRight(data.length - nPast).map(x => Array(x)).toArray.toNDArray
      (x, y)
    }
  }
}