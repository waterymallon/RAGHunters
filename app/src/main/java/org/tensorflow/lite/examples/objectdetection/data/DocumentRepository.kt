package org.tensorflow.lite.examples.objectdetection.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

class DocumentRepository(private val context: Context) {

    private val hscodesDir = "hscodes"

    suspend fun listFiles(): List<String> = withContext(Dispatchers.IO) {
        try {
            context.assets.list(hscodesDir)?.toList() ?: emptyList()
        } catch (e: IOException) {
            e.printStackTrace()
            emptyList()
        }
    }

    suspend fun readFile(fileName: String): String = withContext(Dispatchers.IO) {
        try {
            context.assets.open("$hscodesDir/$fileName").bufferedReader().use { it.readText() }
        } catch (e: IOException) {
            e.printStackTrace()
            "Error reading file: ${e.message}"
        }
    }
}
