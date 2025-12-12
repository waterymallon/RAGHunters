package org.tensorflow.lite.examples.objectdetection.fragments

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.google.gson.Gson
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.objectdetection.ApiService
import org.tensorflow.lite.examples.objectdetection.ChatResponse
import org.tensorflow.lite.examples.objectdetection.SharedViewModel
import org.tensorflow.lite.examples.objectdetection.data.ChatHistoryRepository
import org.tensorflow.lite.examples.objectdetection.data.database.AppDatabase
import org.tensorflow.lite.examples.objectdetection.data.model.ChatMessageEntity

// Define ChatMessage sealed class within ChatbotViewModel's scope
sealed class ChatMessage {
    data class UserQuestion(val question: String) : ChatMessage()
    data class BotResponse(val response: ChatResponse) : ChatMessage()
    data class Error(val message: String) : ChatMessage()
    object Loading : ChatMessage()
}

class ChatbotViewModel(application: Application, private val sharedViewModel: SharedViewModel) : AndroidViewModel(application) {

    private val repository: ChatHistoryRepository
    private val gson = Gson() // For serializing ChatResponse

    // State for the current chat session
    private val _capturedImage = MutableStateFlow<Bitmap?>(null)
    val capturedImage: StateFlow<Bitmap?> = _capturedImage.asStateFlow()

    private val _detectionInfo = MutableStateFlow<String?>(null)
    val detectionInfo: StateFlow<String?> = _detectionInfo.asStateFlow()

    private val _detectedLabels = MutableStateFlow<List<String>?>(null)
    val detectedLabels: StateFlow<List<String>?> = _detectedLabels.asStateFlow()

    private val _chatHistory = MutableStateFlow<List<ChatMessage>>(emptyList())
    val chatHistory: StateFlow<List<ChatMessage>> = _chatHistory.asStateFlow()

    init {
        val chatHistoryDao = AppDatabase.getDatabase(application).chatHistoryDao()
        repository = ChatHistoryRepository(chatHistoryDao, application)

        // Observe new capture initiated from SharedViewModel
        viewModelScope.launch {
            sharedViewModel.newCaptureInitiated.collect { isInitiated ->
                if (isInitiated) {
                    handleNewCapture()
                }
            }
        }
    }

    private fun handleNewCapture() {
        viewModelScope.launch {
            // Save current session if valid and has chat messages
            if (_capturedImage.value != null && !(_detectionInfo.value.isNullOrBlank()) &&
                !(_detectedLabels.value.isNullOrEmpty()) && _chatHistory.value.isNotEmpty()) {

                val currentImage = _capturedImage.value!!
                val currentDetectionInfo = _detectionInfo.value!!
                val currentDetectedLabels = _detectedLabels.value!!
                val currentChatHistory = _chatHistory.value.filter { it !is ChatMessage.Loading }

                val chatMessageEntities = currentChatHistory.map { chatMessage ->
                    when (chatMessage) {
                        is ChatMessage.UserQuestion -> ChatMessageEntity(
                            sessionId = 0,
                            messageType = "USER",
                            content = chatMessage.question,
                            timestamp = System.currentTimeMillis()
                        )
                        is ChatMessage.BotResponse -> ChatMessageEntity(
                            sessionId = 0,
                            messageType = "BOT",
                            content = gson.toJson(chatMessage.response),
                            timestamp = System.currentTimeMillis()
                        )
                        is ChatMessage.Error -> ChatMessageEntity(
                            sessionId = 0,
                            messageType = "ERROR",
                            content = chatMessage.message,
                            timestamp = System.currentTimeMillis()
                        )
                        ChatMessage.Loading -> ChatMessageEntity( // Should not happen due to filter
                            sessionId = 0,
                            messageType = "LOADING",
                            content = "",
                            timestamp = System.currentTimeMillis()
                        )
                    }
                }
                repository.saveChatSession(
                    imageBitmap = currentImage,
                    analysisInfo = currentDetectionInfo,
                    detectedLabels = currentDetectedLabels,
                    chatMessages = chatMessageEntities
                )
            }

            // Clear current state and load new data
            _capturedImage.value = sharedViewModel.capturedImage.value
            _detectionInfo.value = sharedViewModel.detectionInfo.value
            _detectedLabels.value = sharedViewModel.detectedLabels.value
            _chatHistory.value = emptyList()

            // Reset the flag in SharedViewModel
            sharedViewModel.setNewCaptureInitiated(false)
        }
    }

    fun askQuestion(question: String) {
        if (question.isNotBlank() && _chatHistory.value.lastOrNull() !is ChatMessage.Loading) {
            viewModelScope.launch {
                _chatHistory.value = _chatHistory.value + ChatMessage.UserQuestion(question) + ChatMessage.Loading

                val result = ApiService.askQuestion(question)

                _chatHistory.value = _chatHistory.value.dropLast(1) // Remove Loading

                result.onSuccess { response ->
                    _chatHistory.value = _chatHistory.value + ChatMessage.BotResponse(response)
                }.onFailure { error ->
                    _chatHistory.value = _chatHistory.value + ChatMessage.Error("Error: ${error.localizedMessage}")
                }
            }
        }
    }

    // Factory for creating ChatbotViewModel with a constructor that takes Application and SharedViewModel
    class ChatbotViewModelFactory(
        private val application: Application,
        private val sharedViewModel: SharedViewModel
    ) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(ChatbotViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST")
                return ChatbotViewModel(application, sharedViewModel) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}