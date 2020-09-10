package it.unibo.pps18.deeplearning

import it.unibo.pps18.deeplearning.entities.Dataset.TimeSeriesDataset

import org.scalatest.FunSpec

/** Test class to check the correctness of dataset creation. */
class DatasetTest extends FunSpec {

  val data: List[Double] = List(1,2,3,4,5,6,7,8,9)
  val nPast: Int = 2
  val trainTestSplit: Double = 0.5
  val dataset = TimeSeriesDataset(data, nPast, trainTestSplit)

  describe("A dataset") {
    describe("when correct") {
      it("should not be empty") {
        assert(!dataset.isEmpty)
      }
      it("should be composed of 2 elements not empty: training set and test set") {
        assert(!dataset.trainSet._1.data().asDouble().isEmpty && !dataset.trainSet._2.data().asDouble().isEmpty)
        assert(!dataset.testSet._1.data().asDouble().isEmpty && !dataset.testSet._2.data().asDouble().isEmpty)
      }
      it("should be normalized, hence both training set and test set should contain values between 0 and 1") {
        assert(!dataset.trainSet._1.data().asDouble().toList.exists(x => x < 0 || x > 1))
        assert(!dataset.trainSet._2.data().asDouble().toList.exists(x => x < 0 || x > 1))
        assert(!dataset.testSet._1.data().asDouble().toList.exists(x => x < 0 || x > 1))
        assert(!dataset.testSet._2.data().asDouble().toList.exists(x => x < 0 || x > 1))
      }
      it("should be composed of a xTrain and a xTest with each one having a matrix with `nPast` columns") {
        assert(dataset.trainSet._1.columns() == nPast && dataset.testSet._1.columns() == nPast)
      }
      it("should be composed of a yTrain and a yTest with each one having a matrix with one column") {
        assert(dataset.trainSet._2.columns() == 1 && dataset.testSet._2.columns() == 1)
      }
      it("should be composed of a xTrain, yTrain and a xTest, yTest with each one having a number of rows that depends on `trainTestSplit`") {
        assert(dataset.trainSet._1.rows() == (data.length * trainTestSplit - nPast).toInt)
        assert(dataset.trainSet._2.rows() == (data.length * trainTestSplit - nPast).toInt)
        assert(dataset.testSet._1.rows() == (data.length * (1 - trainTestSplit) - nPast + 1).toInt)
        assert(dataset.testSet._2.rows() == (data.length * (1 - trainTestSplit) - nPast + 1).toInt)
      }
    }
  }
}
