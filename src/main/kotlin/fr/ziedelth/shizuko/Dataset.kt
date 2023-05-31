package fr.ziedelth.shizuko

import com.google.gson.Gson
import java.io.File

data class Data(
    val inputs: DoubleArray,
    val outputs: DoubleArray,
)

data class Dataset(
    val trainingSet: Collection<Data>,
    val testSet: Collection<Data>,
) {
    companion object {
        fun load(file: File): Dataset {
            return Gson().fromJson(file.readText(), Dataset::class.java)
        }
    }
}
