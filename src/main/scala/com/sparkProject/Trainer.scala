package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g",
      "spark.debug.maxToStringFields"-> "25"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP3_Spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *./build_and_submit.sh instructions
      ********************************************************************************/

    val basePath= "/home/user/TelecomParistech/TP spark/TP3/" // l'url du dossier de travail , il doit contenir le sous dossier trainingset



    println("0.Reading the data ")


    val input =basePath+"trainingset" //url du dossier training test

    // Chargement du dataset
    var df = spark.read.parquet(input)
    println(df.count())


    /**
      *******************************************************************************
      *  Create pipeline stages  [ preprocessing + model ]
      **********************************************************************************/

    println("1. Create Tokenizer to split the text into words")
    // split text to words
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    println("2. Create stepremover to delete unecessary words")
    // remove unecessary words
    val spremover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    println("3. Create  Count vectorizer TF of TF-IDF")
    // Convert a list of text document into vectors of token counts
    val countvect = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("td")

    println("4. Create IDF stage")
    // It down-weights columns which appear frequently in a corpus.
    val idf = new IDF()
      .setInputCol("td")
      .setOutputCol("tfidf")

    println("5. Convert country column to index ( numerical )")
    // transform string column to numerical
    val index_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")

    println("6. Convert currency column to index ( numerical )")
    // transform string column to numerical
    val index_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")


    println("7. Transform country to OneHotEncoder")
    // transform variable to onehotencoder
    val country_encoder = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("countryVec")

    println("8. Transform currency to OneHotEncoder")
    // transform variable to onehotencoder
    val currency_encoder = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currencyVec")


    println("9. Put the useful features in one column as a vector ")
    //put all features in a single vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "countryVec", "currencyVec"))
      .setOutputCol("features")

    println("10. Define the Classification Model")
    // create the logistic regression model and specifying the parameters
    val model_classifier = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /***************************************************************************************************
      *********************  Set pipeline stages and fiting the model **********************************
      **************************************************************************************************/
    println("11. Put all stages in a pipeline")
    val pipeline = new Pipeline()
        .setStages(Array(tokenizer,
        spremover, countvect, idf,
        index_country, country_encoder, index_currency,currency_encoder,
        assembler, model_classifier))

    println("12. Splitting the data into train and test")

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 99999)


    //pipeline.fit(training)

    /******************************************************************************************************
      * Model parameters with grid-search
      * - Create a grid with parameters values
      * - Use a validation Set to evaluate parameters
      * - Then keep parameters where validation error is minimal
      ******************************************************************************************************/
    println("13. Model Parameters tuning and  estimation")
    // set the gid seach intervals
    val paramGrid = new ParamGridBuilder()
      .addGrid(model_classifier.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(countvect.minDF, Array(55.0, 75.0, 95.0))  //
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1") // we will ue F1-score as a metric

    val trainvalidation = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7) // for the crossvalidation step : 70% training and the rest for test

    val validation_model = trainvalidation.fit(training)

    /*********************************************************************************************************
      * Make predictions on the test-dataset
      ********************************************************************************************************/

    println("14. Making Predictions")

    val df_predictions = validation_model
      .transform(test)
      .select("features", "final_status", "predictions", "raw_predictions")



    /********************************************************************************************************
      * Evaluate F1 Metric on Test-data predictions
      * Print counts of prediction dataframe
      ********************************************************************************************************/

    println("15. Evaluating the model")

    val metric = evaluator.evaluate(df_predictions)

    println("The F1 score = "+ metric)
    
    println("The Confusion matrix :")
    df_predictions.groupBy("final_status", "predictions").count.show()


    println("16. Saving  the model")

    validation_model.write.overwrite().save(basePath +"sample-model")
  }


}

