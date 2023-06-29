package fr.ziedelth.shizuko

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.util.*
import kotlin.math.pow

data class Matrix(
    var rows: Int,
    private var columns: Int,
    var data: DoubleArray = DoubleArray(rows * columns),
) {
    fun get(x: Int, y: Int): Double {
        return data[x * columns + y]
    }

    fun set(x: Int, y: Int, value: Double) {
        data[x * columns + y] = value
    }

    fun map(function: (Double, Int, Int) -> Double): Matrix {
        for (index in data.indices) {
            val x = (index * (1.0 / columns)).toInt()
            val y = index % columns
            data[index] = function(data[index], x, y)
        }

        return this
    }

    fun randomize(random: Random, min: Double = -1.0, max: Double = 1.0): Matrix {
        return map { _, _, _ -> random.nextDouble() * (max - min) + min }
    }

    fun add(other: Matrix): Matrix {
        for (index in data.indices) {
            data[index] += other.data[index]
        }

        return this
    }

    fun subtract(other: Matrix): Matrix {
        for (index in data.indices) {
            data[index] -= other.data[index]
        }

        return this
    }

    fun transpose(): Matrix {
        val result = Matrix(columns, rows)

        for (x in 0 until rows) {
            for (y in 0 until columns) {
                result.set(y, x, get(x, y))
            }
        }

        return result
    }

    fun elementMult(other: Matrix): Matrix {
        for (index in data.indices) {
            data[index] *= other.data[index]
        }

        return this
    }

    fun multiply(other: Matrix): Matrix {
        val columns = other.columns
        val result = DoubleArray(rows * columns)

        if (rows * columns <= 512) {
            for (x in 0 until rows) {
                for (y in 0 until columns) {
                    var sum = 0.0
                    for (i in 0 until this.columns) {
                        sum += get(x, i) * other.get(i, y)
                    }
                    result[x * columns + y] = sum
                }
            }
        } else {
            runBlocking {
                coroutineScope {
                    for (x in 0 until rows) {
                        launch {
                            for (y in 0 until columns) {
                                var sum = 0.0
                                for (i in 0 until this@Matrix.columns) {
                                    sum += get(x, i) * other.get(i, y)
                                }
                                result[x * columns + y] = sum
                            }
                        }
                    }
                }
            }
        }

        return Matrix(rows, columns, result)
    }

    fun scale(scalar: Double): Matrix {
        for (index in data.indices) {
            data[index] *= scalar
        }

        return this
    }

    fun toArray(): DoubleArray {
        val result = DoubleArray(rows * columns)
        System.arraycopy(data, 0, result, 0, rows * columns)
        return result
    }

    fun copy(): Matrix {
        return Matrix(rows, columns, data.copyOf())
    }

    fun pow(power: Double): Matrix {
        for (index in data.indices) {
            data[index] = data[index].pow(power)
        }

        return this
    }

    fun mean(): Double {
        return data.average()
    }

    fun sum(): Double {
        return data.sum()
    }

    companion object {
        fun fromArray(array: DoubleArray): Matrix {
            val result = Matrix(array.size, 1)
            System.arraycopy(array, 0, result.data, 0, array.size)
            return result
        }
    }
}
