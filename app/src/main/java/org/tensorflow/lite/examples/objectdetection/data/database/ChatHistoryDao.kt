package org.tensorflow.lite.examples.objectdetection.data.database

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Transaction
import kotlinx.coroutines.flow.Flow
import org.tensorflow.lite.examples.objectdetection.data.model.ChatMessageEntity
import org.tensorflow.lite.examples.objectdetection.data.model.ChatSession

@Dao
interface ChatHistoryDao {
    @Insert
    suspend fun insertSession(session: ChatSession): Long

    @Insert
    suspend fun insertMessages(messages: List<ChatMessageEntity>)

    @Transaction
    suspend fun insertSessionWithMessages(session: ChatSession, messages: List<ChatMessageEntity>) {
        val sessionId = insertSession(session)
        val messagesWithSessionId = messages.map { it.copy(sessionId = sessionId) }
        insertMessages(messagesWithSessionId)
    }

    @Query("SELECT * FROM chat_sessions ORDER BY timestamp DESC")
    fun getAllSessions(): Flow<List<ChatSession>>

    @Query("SELECT * FROM chat_sessions WHERE id = :sessionId")
    suspend fun getSessionById(sessionId: Long): ChatSession?

    @Query("SELECT * FROM chat_messages WHERE sessionId = :sessionId ORDER BY timestamp ASC")
    fun getMessagesForSession(sessionId: Long): Flow<List<ChatMessageEntity>>

    @Query("SELECT content FROM chat_messages WHERE sessionId = :sessionId AND messageType = 'BOT' ORDER BY timestamp ASC")
    suspend fun getAllBotMessageContents(sessionId: Long): List<String>

    @Query("SELECT content FROM chat_messages WHERE messageType = 'BOT' ORDER BY timestamp ASC")
    suspend fun getAllBotMessageContentsForAllSessions(): List<String>

    @Delete
    suspend fun deleteSession(session: ChatSession)

    @Query("DELETE FROM chat_sessions")
    suspend fun clearAllSessions()
}
