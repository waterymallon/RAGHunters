package org.tensorflow.lite.examples.objectdetection.fragments

import android.app.Application
import androidx.lifecycle.*
import com.google.gson.Gson
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.objectdetection.ChatResponse
import org.tensorflow.lite.examples.objectdetection.data.ChatHistoryRepository
import org.tensorflow.lite.examples.objectdetection.data.database.AppDatabase
import org.tensorflow.lite.examples.objectdetection.data.model.ChatSession

import kotlinx.coroutines.ExperimentalCoroutinesApi

@OptIn(ExperimentalCoroutinesApi::class)
class HistoryViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: ChatHistoryRepository
    private val gson = Gson()

    val allSessions: LiveData<List<HistoryListItem>>

    init {
        val chatHistoryDao = AppDatabase.getDatabase(application).chatHistoryDao()
        repository = ChatHistoryRepository(chatHistoryDao, application)

        allSessions = repository.getAllChatSessions().flatMapLatest { sessions ->
            flow {
                val listItems = sessions.map { session ->
                    val botMessageContents = repository.getAllBotMessageContents(session.id)
                    val allTariffs = mutableListOf<org.tensorflow.lite.examples.objectdetection.TariffInfo>()

                    botMessageContents.forEach { content ->
                        try {
                            val chatResponse = gson.fromJson(content, ChatResponse::class.java)
                            chatResponse.tariffInfo?.let { allTariffs.addAll(it) }
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }

                    val tariffInfoTitle = allTariffs
                        .take(3)
                        .mapNotNull { tariff ->
                            if (tariff.itemNumber != null && tariff.itemName != null && tariff.rate != null) {
                                "${tariff.itemNumber} ${tariff.itemName} ${tariff.rate}"
                            } else {
                                null
                            }
                        }
                        .joinToString("\n")
                        .takeIf { it.isNotBlank() }

                    HistoryListItem(session, allTariffs, tariffInfoTitle)
                }
                emit(listItems)
            }
        }.asLiveData()
    }

    fun deleteSession(item: HistoryListItem) {
        viewModelScope.launch {
            repository.deleteSession(item.session)
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