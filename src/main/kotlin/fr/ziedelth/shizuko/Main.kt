package fr.ziedelth.shizuko

import java.io.File

fun main() {
    val dataset = Dataset.load(File("dataset.json"))
    println("Training set size: ${dataset.trainingSet.size}")
    println("Test set size: ${dataset.testSet.size}")

    val neuralNetwork = NeuralNetwork(24 * 24, 3, 64, 4)
    dataset.train(neuralNetwork, 32, 200)

    neuralNetwork.save(File("neural-network.json"))
    Parallelism.stop()
}