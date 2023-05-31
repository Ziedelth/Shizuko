package fr.ziedelth.shizuko

import fr.ziedelth.shizuko.Parallelism.parallelForEachChunked
import java.io.File
import kotlin.system.measureTimeMillis

fun bench() {
    val dataset = arrayOf(
        doubleArrayOf(0.0, 0.0) to doubleArrayOf(0.0),
        doubleArrayOf(0.0, 1.0) to doubleArrayOf(1.0),
        doubleArrayOf(1.0, 0.0) to doubleArrayOf(1.0),
        doubleArrayOf(1.0, 1.0) to doubleArrayOf(0.0),
    ).toList()

    val neuralNetwork = NeuralNetwork(2, 1, 4, 1, 0.7)

    (1..10_000).forEach { _ ->
        dataset.forEach { (input, target) ->
            neuralNetwork.train(input, target)
        }
    }

    dataset.forEach { (input, target) ->
        println("Input: ${input.contentToString()}, Output: ${neuralNetwork.feedForward(input).contentToString()}, Target: ${target.contentToString()}")
    }

    neuralNetwork.save(File("neural-network.json"))
}

fun main() {
//    val time = measureTimeMillis { bench() }
//    println("Bench took $time ms")
//    return

    val dataset = Dataset.load(File("dataset.json"))
    println("Training set size: ${dataset.trainingSet.size}")
    println("Test set size: ${dataset.testSet.size}")

    val neuralNetwork = NeuralNetwork(24 * 24, 3, 64, 4)
    val times = mutableListOf<Long>()

    (1..50).forEach {
        val time = measureTimeMillis {
            dataset.trainingSet.parallelForEachChunked { data ->
                neuralNetwork.train(data.inputs, data.outputs)
            }
        }

        times.add(time)
        println("Epoch $it took $time ms")
    }
    println("Average epoch time: ${times.average()} ms")

    neuralNetwork.save(File("neural-network.json"))
    Parallelism.stop()
}