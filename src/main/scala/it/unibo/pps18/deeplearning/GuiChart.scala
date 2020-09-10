package it.unibo.pps18.deeplearning

import it.unibo.pps18.deeplearning.utils.CsvReader.getDataFromCsv
import it.unibo.pps18.deeplearning.ai.NeuralModel.getForecasting
import it.unibo.pps18.deeplearning.entities.Dataset.TimeSeriesDataset

import scalafx.application.JFXApp
import scalafx.application.JFXApp.PrimaryStage
import scalafx.collections.ObservableBuffer
import scalafx.scene.Scene
import scalafx.scene.chart.{LineChart, NumberAxis, XYChart}

/** ScalaFX application to create a chart in order to visualize the forecasting of the deep learning model. */
object GuiChart extends JFXApp {

  // Getting data from the file.
  val csvFilePath: String = "resources/AirlinePassengers.csv"
  val data: List[Double] = getDataFromCsv(csvFilePath)

  // Window width.
  val nPast: Int = 11
  // Split percentage to create the training set and the test set from data.
  val trainTestSplit: Double = 0.7
  // Future values to predict.
  val elementsToPredict: Int = 12

  // Dataset creation.
  val dataset = TimeSeriesDataset(data, nPast, trainTestSplit)

  // Getting the predictions on the test set and the forecasting of the deep learning model.
  val (predictions, forecasting) = getForecasting(dataset, elementsToPredict)

  // Defining the axes of the chart with their respective label.
  val xAxis: NumberAxis = NumberAxis()
  xAxis.label = "Months"
  val yAxis: NumberAxis = NumberAxis()
  yAxis.label = "Passengers"

  // Creating the chart.
  val lineChart: LineChart[Number, Number] = LineChart(xAxis, yAxis)
  lineChart.title = "International Airline Passengers"

  /** Draw a data series on the chart.
   *
   *  @param dataToDraw the data to add to the chart
   *  @param initialPosition the initial position on the x-axis
   *  @param label the description of the series
   */
  def drawSeriesToChart(dataToDraw: List[Double], initialPosition: Int, label: String): Unit = {
    val data = ObservableBuffer(dataToDraw.zipWithIndex.map({
      case (x, count) => XYChart.Data[Number, Number](count + initialPosition, x)
    }))
    lineChart.getData.add(XYChart.Series[Number, Number](label, data))
  }

  // Drawing the original data on the chart.
  drawSeriesToChart(data.toList, 1, "Data")
  // Drawing the predictions on the chart.
  drawSeriesToChart(predictions, (data.length * trainTestSplit).toInt + nPast + 1, "Predictions")
  // Drawing the forecasting on the chart.
  drawSeriesToChart(forecasting, data.length, "Forecasting")

  // Creating the stage to plot the chart.
  stage = new PrimaryStage {
    title = "Time Series Forecasting with DeepLearning.scala"
    scene = new Scene(800, 600) {
      root = lineChart
    }
  }
}
