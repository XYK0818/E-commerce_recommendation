//export SPARK_MAJOR_VERSION=2
//spark-shell --packages com.databricks:spark-csv_2.11:1.2.                                                                                                                                                             0

import org.apache.spark.sql._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
//http://help.sentiment140.com/for-students
val tweetsDF = sqlContext.read.format("com.databricks.spark.csv").option("header", "false").load("hdfs:///tmp/twitter/train.csv").toDF("polarity", "id", "date", "query", "user", "tweets")

val tweetsRDD = tweetsDF.select("polarity", "tweets").rdd


val labeledRDD = tweetsRDD.map{line =>
val polarity = line(0).toString.toDouble
val tweet = line(1).toString
//https://spark.apache.org/docs/1.6.2/mllib-feature-extraction.html
val hashingTF = new HashingTF()
val features = hashingTF.transform(tweet.toLowerCase().replaceAll("\n", "").split("\\W+").filter(_.matches("^[a-zA-Z]+$")))
LabeledPoint(polarity, features)
}

val naiveBayesModel = NaiveBayes.train(labeledRDD, lambda = 1.0, modelType = "multinomial")
naiveBayesModel.save(sc, "hdfs:///tmp/twitter/twitter_model")

val naiveBayesModel = NaiveBayesModel.load(sc, "hdfs:///tmp/twitter/twitter_model")

val test_tweetsDF = sqlContext.read.format("com.databricks.spark.csv").option("header", "false").load("hdfs:///tmp/twitter/test.csv").toDF("polarity", "id", "date", "query", "user", "tweets")
val test_tweetsRDD = test_tweetsDF.select("polarity", "tweets").rdd

val test_labeledRDD = test_tweetsRDD.map{line =>
val polarity = line(0).toString.toDouble
val tweet = line(1).toString
val hashingTF = new HashingTF()
val features = hashingTF.transform(tweet.toLowerCase().replaceAll("\n", "").split("\\W+").filter(_.matches("^[a-zA-Z]+$")))
(polarity, naiveBayesModel.predict(features))
}
val accuracy  = test_labeledRDD.filter(line => line._1 == line._2).count().toDouble / test_labeledRDD.count()

