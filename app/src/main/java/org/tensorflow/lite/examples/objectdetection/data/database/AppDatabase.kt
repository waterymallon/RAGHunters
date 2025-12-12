package org.tensorflow.lite.examples.objectdetection.data.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import org.tensorflow.lite.examples.objectdetection.data.model.ChatMessageEntity
import org.tensorflow.lite.examples.objectdetection.data.model.ChatSession

@Database(entities = [ChatSession::class, ChatMessageEntity::class], version = 1, exportSchema = false)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun chatHistoryDao(): ChatHistoryDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "chat_history_db"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
