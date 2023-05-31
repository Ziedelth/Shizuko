package fr.ziedelth.shizuko

import java.util.Random

data class Matrix(
    val rows: Int,
    val columns: Int,
    private var data: DoubleArray = DoubleArray(rows * columns)
) {
    @Transient
    private val indices = data.indices
    @Transient
    private val random = Random()

    fun get(row: Int, column: Int): Double {
        return data[row * columns + column]
    }

    fun set(row: Int, column: Int, value: Double) {
        data[row * columns + column] = value
    }

    fun map(function: (Double, Int, Int) -> Double): Matrix {
        indices.forEach { index ->
            val x = index % columns
            val y = index / columns
            set(y, x, function(get(y, x), y, x))
        }

        return this
    }

    fun randomize(min: Double = -1.0, max: Double = 1.0): Matrix {
        return map { _, _, _ -> random.nextDouble() * (max - min) + min }
    }

    fun add(other: Matrix): Matrix {
        return map { `val`, x, y -> `val` + other.get(x, y) }
    }

    fun subtract(other: Matrix): Matrix {
        return map { `val`, x, y -> `val` - other.get(x, y) }
    }

    fun transpose(): Matrix {
        val result = Matrix(columns, rows)
        result.map { _, x, y -> get(y, x) }
        return result
    }

    fun elementMult(other: Matrix): Matrix {
        return map { `val`, x, y -> `val` * other.get(x, y) }
    }

    fun multiply(other: Matrix): Matrix {
        return Matrix(rows, other.columns).map { _, x, y ->
            var sum = 0.0
            (0 until columns).forEach { i -> sum += get(x, i) * other.get(i, y) }
            sum
        }
    }

    fun scale(scalar: Double): Matrix {
        return map { `val`, _, _ -> `val` * scalar }
    }

    fun toArray(): DoubleArray {
        val result = DoubleArray(rows * columns)
        indices.forEach { index -> result[index] = data[index] }
        return result
    }

    companion object {
        fun fromArray(array: DoubleArray): Matrix {
            val result = Matrix(array.size, 1)
            array.indices.forEach { row -> result.set(row, 0, array[row]) }
            return result
        }
    }
}
