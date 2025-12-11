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
    private val API_URL = "${BuildConfig.SERVER_IP}/ask"

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        encodeDefaults = true
    }

    suspend fun askQuestion(question: String): Result<ChatResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val url = URL(API_URL)
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                connection.connectTimeout = 30000 // ngrok 지연 고려 30초
                connection.readTimeout = 30000
                connection.doOutput = true
                connection.doInput = true

                // JSON Payload
                val payload = "{\"question\":\"$question\"}"

                val outputStreamWriter = OutputStreamWriter(connection.outputStream, "UTF-8")
                outputStreamWriter.write(payload)
                outputStreamWriter.flush()
                outputStreamWriter.close()

                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val inputStream = BufferedReader(InputStreamReader(connection.inputStream))
                    val responseText = inputStream.readText()
                    inputStream.close()

                    val chatResponse = json.decodeFromString<ChatResponse>(responseText)
                    Result.success(chatResponse)
                } else {
                    Result.failure(Exception("서버 응답 실패 : $responseCode"))
                }
            } catch (e: Exception) {
                e.printStackTrace()
                Result.failure(e)
            }
        }
    }
}