package fr.ziedelth.shizuko

import java.io.File

fun main() {
//    val neuralNetwork = NeuralNetwork(2, 3, 8, 1, 0.1)
//
//    for (i in 0..100_000) {
//        neuralNetwork.train(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0))
//        neuralNetwork.train(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0))
//        neuralNetwork.train(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0))
//        neuralNetwork.train(doubleArrayOf(1.0, 1.0), doubleArrayOf(0.0))
//    }
//
//    println(neuralNetwork.feedForward(doubleArrayOf(0.0, 0.0)).contentToString())
//    println(neuralNetwork.feedForward(doubleArrayOf(0.0, 1.0)).contentToString())
//    println(neuralNetwork.feedForward(doubleArrayOf(1.0, 0.0)).contentToString())
//    println(neuralNetwork.feedForward(doubleArrayOf(1.0, 1.0)).contentToString())

    val dataset = Dataset.load(File("dataset.json"))
    println("Training set size: ${dataset.trainingSet.size}")
    println("Test set size: ${dataset.testSet.size}")

    val neuralNetwork = NeuralNetwork(24 * 24, 3, 32, 4)
    dataset.train(neuralNetwork, 500)

    neuralNetwork.save(File("neural-network.json"))
}