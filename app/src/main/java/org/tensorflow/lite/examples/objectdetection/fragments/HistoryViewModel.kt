package org.tensorflow.lite.examples.objectdetection.fragments

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.asLiveData
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.objectdetection.data.ChatHistoryRepository
import org.tensorflow.lite.examples.objectdetection.data.database.AppDatabase
import org.tensorflow.lite.examples.objectdetection.data.model.ChatSession

class HistoryViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: ChatHistoryRepository

    init {
        val chatHistoryDao = AppDatabase.getDatabase(application).chatHistoryDao()
        repository = ChatHistoryRepository(chatHistoryDao, application)
    }

    val allSessions = repository.getAllChatSessions().asLiveData()

    fun deleteSession(session: ChatSession) {
        viewModelScope.launch {
            repository.deleteSession(session)
        }
    }

    // Factory for creating HistoryViewModel with a constructor that takes an Application
    class HistoryViewModelFactory(private val application: Application) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(HistoryViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST")
                return HistoryViewModel(application) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}
