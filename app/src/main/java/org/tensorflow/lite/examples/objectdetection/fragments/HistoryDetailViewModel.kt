package org.tensorflow.lite.examples.objectdetection.fragments

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.asLiveData
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.objectdetection.data.ChatHistoryRepository
import org.tensorflow.lite.examples.objectdetection.data.database.AppDatabase
import org.tensorflow.lite.examples.objectdetection.data.model.ChatMessageEntity
import org.tensorflow.lite.examples.objectdetection.data.model.ChatSession

class HistoryDetailViewModel(application: Application, private val sessionId: Long) : AndroidViewModel(application) {

    private val repository: ChatHistoryRepository

    private val _session = MutableLiveData<ChatSession?>()
    val session: LiveData<ChatSession?> get() = _session

    val messages: LiveData<List<ChatMessageEntity>>

    init {
        val chatHistoryDao = AppDatabase.getDatabase(application).chatHistoryDao()
        repository = ChatHistoryRepository(chatHistoryDao, application)

        viewModelScope.launch {
            _session.value = repository.getChatSessionById(sessionId)
        }
        messages = repository.getChatMessagesForSession(sessionId).asLiveData()
    }

    class HistoryDetailViewModelFactory(private val application: Application, private val sessionId: Long) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(HistoryDetailViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST")
                return HistoryDetailViewModel(application, sessionId) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}
