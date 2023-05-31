package fr.ziedelth.shizuko

import kotlin.math.exp

private interface IFunction {
    fun apply(matrix: Matrix): Matrix
    fun applyDerivative(matrix: Matrix): Matrix
}

private class Sigmoid : IFunction {
    override fun apply(matrix: Matrix): Matrix {
        return matrix.map { `val`, _, _ -> 1 / (1 + exp(-`val`)) }
    }

    override fun applyDerivative(matrix: Matrix): Matrix {
        return matrix.map { `val`, _, _ -> `val` * (1 - `val`) }
    }
}

private val functions = mapOf(
    "sigmoid" to Sigmoid(),
)

data class NeuralNetwork(
    private val inputs: Int,
    private val hiddenLayers: Int,
    private val hidden: Int,
    private val outputs: Int,
    private val learningRate: Double = 0.001,
    private val activationFunction: String = "sigmoid",
) {
    @Transient
    private val function: IFunction = functions[activationFunction] ?: error("Unknown activation function: $activationFunction")

    private val weights: Array<Matrix> = Array(hiddenLayers + 1) {
        when (it) {
            0 -> {
                Matrix(hidden, inputs).randomize()
            }
            hiddenLayers -> {
                Matrix(outputs, hidden).randomize()
            }
            else -> {
                Matrix(hidden, hidden).randomize()
            }
        }
    }
    private val biases: Array<Matrix> = Array(hiddenLayers + 1) {
        Matrix(if (it == hiddenLayers) outputs else hidden, 1).randomize()
    }

    private fun calculateLayer(weights: Matrix, biases: Matrix, inputs: Matrix): Matrix {
        return function.apply(weights.multiply(inputs).add(biases))
    }

    private fun calculateGradient(output: Matrix, target: Matrix): Matrix {
        return function.applyDerivative(output).elementMult(target.subtract(output)).scale(learningRate)
    }

    private fun calculateDeltas(gradient: Matrix, previousOutput: Matrix): Matrix {
        return gradient.multiply(previousOutput.transpose())
    }

    fun feedForward(inputArray: DoubleArray): DoubleArray {
        var inputs = Matrix.fromArray(inputArray)
        (0 until hiddenLayers + 1).forEach { inputs = calculateLayer(weights[it], biases[it], inputs) }
        return inputs.toArray()
    }

    fun train(inputArray: DoubleArray, targetArray: DoubleArray) {
        var inputs = Matrix.fromArray(inputArray)
        var targets = Matrix.fromArray(targetArray)

        val layers = arrayOfNulls<Matrix>(hiddenLayers + 2)
        layers[0] = inputs

        (1 until hiddenLayers + 2).forEach {
            layers[it] = calculateLayer(weights[it - 1], biases[it - 1], inputs)
            inputs = layers[it]!!
        }

        (hiddenLayers + 1 downTo 1).forEach {
            val errors = targets.subtract(layers[it]!!)
            val gradients = calculateGradient(layers[it]!!, errors)
            val deltas = calculateDeltas(gradients, layers[it - 1]!!)
            biases[it - 1] = biases[it - 1].add(gradients)
            weights[it - 1] = weights[it - 1].add(deltas)
            val previousError = weights[it - 1].transpose().multiply(errors)
            targets = previousError
        }
    }
}
