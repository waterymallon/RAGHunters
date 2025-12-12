package org.tensorflow.lite.examples.objectdetection.data

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import org.tensorflow.lite.examples.objectdetection.data.database.ChatHistoryDao
import org.tensorflow.lite.examples.objectdetection.data.model.ChatMessageEntity
import org.tensorflow.lite.examples.objectdetection.data.model.ChatSession
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class ChatHistoryRepository(
    private val chatHistoryDao: ChatHistoryDao,
    private val context: Context
) {

    suspend fun saveChatSession(
        imageBitmap: Bitmap?,
        analysisInfo: String?,
        detectedLabels: List<String>?,
        chatMessages: List<ChatMessageEntity>
    ): Long = withContext(Dispatchers.IO) {
        val imagePath = imageBitmap?.let { saveBitmapToInternalStorage(it) }
        val session = ChatSession(
            imagePath = imagePath,
            analysisInfo = analysisInfo,
            detectedLabels = detectedLabels,
            timestamp = System.currentTimeMillis()
        )
        val sessionId = chatHistoryDao.insertSession(session)

        val messagesWithSessionId = chatMessages.map { it.copy(sessionId = sessionId) }
        chatHistoryDao.insertMessages(messagesWithSessionId)
        sessionId
    }

    fun getAllChatSessions(): Flow<List<ChatSession>> {
        return chatHistoryDao.getAllSessions()
    }

    suspend fun getChatSessionById(sessionId: Long): ChatSession? {
        return chatHistoryDao.getSessionById(sessionId)
    }

    fun getChatMessagesForSession(sessionId: Long): Flow<List<ChatMessageEntity>> {
        return chatHistoryDao.getMessagesForSession(sessionId)
    }

    suspend fun getAllBotMessageContents(sessionId: Long): List<String> {
        return chatHistoryDao.getAllBotMessageContents(sessionId)
    }

    suspend fun getAllBotMessageContents(): List<String> {
        return chatHistoryDao.getAllBotMessageContentsForAllSessions()
    }

    suspend fun deleteSession(session: ChatSession) = withContext(Dispatchers.IO) {
        // Delete the image file from internal storage if it exists
        session.imagePath?.let { path ->
            try {
                val imageFile = File(path)
                if (imageFile.exists()) {
                    imageFile.delete()
                }
            } catch (e: Exception) {
                e.printStackTrace() // Log the error
            }
        }

        // Delete the session from the database
        chatHistoryDao.deleteSession(session)
    }

    private fun saveBitmapToInternalStorage(bitmap: Bitmap): String {
        val filename = "chat_session_${System.currentTimeMillis()}.png"
        val file = File(context.filesDir, filename)
        try {
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out) // bmp is your Bitmap instance
            }
            return file.absolutePath
        } catch (e: IOException) {
            e.printStackTrace()
            // Handle error, maybe return a default path or throw
            return ""
        }
    }

    suspend fun clearAllSessions() {
        chatHistoryDao.clearAllSessions()
        // Optionally, also delete image files from internal storage
        withContext(Dispatchers.IO) {
            context.filesDir.listFiles { _, name -> name.startsWith("chat_session_") }?.forEach {
                it.delete()
            }
        }
    }
}
