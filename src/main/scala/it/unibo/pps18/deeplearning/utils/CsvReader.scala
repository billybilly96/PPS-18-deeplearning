package it.unibo.pps18.deeplearning.utils

import scala.collection.mutable.ListBuffer
import scala.util.{Failure, Success, Try}

/** Utility object for csv file processing. */
object CsvReader {

  /** Close a resource after using it.
   *
   *  @param A the resource to close after use
   *  @param B the function from A to B to execute that uses the resource
   *  @return the success or failure of the function with the final closure of the resource.
   */
  private def using[A <: {def close(): Unit}, B](resource: A)(f: A => B): B = {
    Try(f(resource)) match {
      case Success(result) => {
        resource.close()
        result
      }
      case Failure(e) => {
        resource.close()
        throw e
      }
    }
  }

  /** Get data from a csv file.
   *
   *  @param path the csv file path
   *  @return the data values.
   */
  def getDataFromCsv(path: String): List[Double] = {
    val rows = ListBuffer[List[String]]()
    using(io.Source.fromFile(path)) { source =>
      for (line <- source.getLines) {
        rows += line.split(",").map(_.trim).toList
      }
      // Removing the header.
      rows.remove(0)
    }
    // Getting the second column values from the csv.
    val data = for (row <- rows) yield row(1)
    data.toList.map(_.toDouble)
  }

}
