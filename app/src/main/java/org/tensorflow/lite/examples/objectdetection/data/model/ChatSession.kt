package org.tensorflow.lite.examples.objectdetection.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "chat_sessions")
data class ChatSession(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val imagePath: String?, // Path to the saved captured image
    val analysisInfo: String?, // The "📊 분석 리포트" text
    val detectedLabels: List<String>?, // Handled by TypeConverter
    val timestamp: Long
)
