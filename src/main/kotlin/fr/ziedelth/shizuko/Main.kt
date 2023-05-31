package fr.ziedelth.shizuko

fun main() {
    val dataset = arrayOf(
        doubleArrayOf(0.0, 0.0) to doubleArrayOf(0.0),
        doubleArrayOf(0.0, 1.0) to doubleArrayOf(1.0),
        doubleArrayOf(1.0, 0.0) to doubleArrayOf(1.0),
        doubleArrayOf(1.0, 1.0) to doubleArrayOf(0.0),
    )

    val neuralNetwork = NeuralNetwork(2, 1, 4, 1, 0.1)

    dataset.forEach { (input, target) ->
        println("Input: ${input.contentToString()} Target: ${target.contentToString()} Output: ${neuralNetwork.feedForward(input).contentToString()}")
    }

    println()

    (1..100_000).forEach { _ ->
        dataset.toList().shuffled().forEach { (input, target) ->
            neuralNetwork.train(input, target)
        }
    }

    dataset.forEach { (input, target) ->
        println("Input: ${input.contentToString()} Target: ${target.contentToString()} Output: ${neuralNetwork.feedForward(input).contentToString()}")
    }

    // Print the memory usage
    println("Memory usage: ${(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024)} MB")
}