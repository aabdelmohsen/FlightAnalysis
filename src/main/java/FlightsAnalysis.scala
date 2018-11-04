import java.io._
import scala.io.Source
import scala.math.random
import Numeric.Implicits._
import org.apache.spark._
import org.apache.spark.sql.{ DataFrame, SQLContext }
import au.com.bytecode.opencsv.CSVParser
import org.apache.spark.sql.functions._
import java.util._
import org.apache.spark.mllib.linalg.{ Vectors, Vector }
import org.apache.spark.mllib.stat.Statistics
import collection.breakOut
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

object FlightsAnalysis extends App {

  val confg = new SparkConf().setAppName("Flights Distance Analysis").setMaster("local[2]")
  val sc = new SparkContext(confg)

  //Read the flights data from HDFS
  val csv = sc.textFile("/usr/cloudera/dataset/input/flights.csv")
  val headerAndRows = csv.map(line => line.split(",").map(_.trim))
  // Removing the headers
  val header = headerAndRows.first
  val flights = headerAndRows.filter(_(0) != header(0))

  // Take 0.25 Sample without replacement
  val sampleWithoutReplacement = flights.sample(false, 0.25)

  // calculate the mean and average for each sample
  def calculateMeanAndVariance(input: RDD[Array[String]]): RDD[(String, Double, Double)] = {
    val groupByAirLines = flights.map(row => (row(4), row(17).toDouble)).groupByKey()
    val averageByDistance = groupByAirLines
      .map(x => (x._1, (x._2.count(_ => true), x._2.reduce(_ + _).toDouble), x._2))
      .map(x => (x._1, x._2._2 / x._2._1, x._3))
    val meanAndVariance = averageByDistance
      .map(e => (e._1, e._2, e._3.map(v => scala.math.pow(v - e._2, 2).toDouble)))
      .map(e => (e._1, "%.3f".format(e._2).toDouble, e._3.reduce(_ + _) / e._3.size))
    return meanAndVariance;
  }

  // Take 1000 sample with replacement
  val sampleWithReplacement = (1 to 100).map(x => calculateMeanAndVariance(sampleWithoutReplacement.sample(true, 1))).reduce(_ union _)

  // Create Pairs of (key, data[average, variance])
  val averageVariancePairs = sampleWithReplacement
    .map { case (key, avg, variance) => ((key, avg), variance) }.reduceByKey(_ + _)
    .map { case ((key, avg), variance) => (key, (avg, variance)) }.groupByKey
    .map({ case (key, data) => (key, data.unzip) })

  val finalMeanForSamples = averageVariancePairs
    .map(f => ("" + f._1 + ",\t" + f._2._1.reduce(_ + _) / 100) + ",\t" + "%.3f".format(f._2._2.reduce(_ + _) / 100).toDouble)

  val headerRDD = sc.parallelize(Seq("Category,\t Mean,\t Variance"))
  val finalResult = headerRDD ++ finalMeanForSamples
  finalResult.coalesce(1, true).saveAsTextFile("/usr/cloudera/dataset/output")

}

