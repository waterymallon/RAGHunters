package org.tensorflow.lite.examples.objectdetection

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

object ApiService {

    // NOTE: 10.0.2.2 is the special IP address for the host machine's localhost
    // when running the app in the Android emulator.
    private val API_URL = "http://${BuildConfig.SERVER_IP}:5000/ask"

    // Configure a lenient Json parser
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    suspend fun askQuestion(question: String): Result<ChatResponse> {
        // Switch to the I/O dispatcher for the network call
        return withContext(Dispatchers.IO) {
            try {
                val url = URL(API_URL)
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                connection.doOutput = true
                connection.doInput = true

                // Create JSON payload
                val payload = "{\"question\":\"$question\"}"

                // Send request
                val outputStreamWriter = OutputStreamWriter(connection.outputStream, "UTF-8")
                outputStreamWriter.write(payload)
                outputStreamWriter.flush()
                outputStreamWriter.close()

                // Check response code
                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    // Read response
                    val inputStream = BufferedReader(InputStreamReader(connection.inputStream))
                    val response = inputStream.readText()
                    inputStream.close()

                    // Decode JSON response to our data class
                    val chatResponse = json.decodeFromString<ChatResponse>(response)
                    Result.success(chatResponse)
                } else {
                    Result.failure(Exception("Failed to get response. HTTP Code: $responseCode"))
                }
            } catch (e: Exception) {
                e.printStackTrace()
                Result.failure(e)
            }
        }
    }
}
