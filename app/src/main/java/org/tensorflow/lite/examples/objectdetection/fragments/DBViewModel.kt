package org.tensorflow.lite.examples.objectdetection.fragments

import android.app.Application
import android.text.Spannable
import android.text.SpannableString
import android.text.style.BackgroundColorSpan
import androidx.lifecycle.*
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.objectdetection.data.DocumentRepository

class DBViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = DocumentRepository(application)

    private val _allFiles = MutableLiveData<List<String>>()
    private val _filteredFiles = MutableLiveData<List<String>>()
    val displayedFiles: LiveData<List<String>> = _filteredFiles

    private val _fileContent = MutableLiveData<Spannable?>()
    val fileContent: LiveData<Spannable?> = _fileContent

    private val _uiState = MutableLiveData<UiState>(UiState.ShowList)
    val uiState: LiveData<UiState> = _uiState

    init {
        viewModelScope.launch {
            val files = repository.listFiles()
            _allFiles.postValue(files)
            _filteredFiles.postValue(files)
        }
    }

    fun performSearch(query: String?) {
        if (query.isNullOrBlank()) {
            _filteredFiles.value = _allFiles.value
            showList()
            return
        }

        // Normalize 6-digit search query
        val normalizedQuery = query.replace(".", "").replace("-", "")
        if (normalizedQuery.length == 6 && normalizedQuery.all { it.isDigit() }) {
            handleSixDigitSearch(normalizedQuery)
            return
        }

        // Handle 4-digit search query specially (find files that start with these digits)
        if (query.length == 4 && query.all { it.isDigit() }) {
            val matchingFiles = _allFiles.value?.filter { it.startsWith(query) }
            if (matchingFiles?.size == 1) {
                loadFile(matchingFiles.first()) // Auto-load if only one match
                return
            } else {
                _filteredFiles.value = matchingFiles ?: emptyList() // Show filtered list if multiple or none
                showList()
                return
            }
        }

        // Normal title search for any other query
        _filteredFiles.value = _allFiles.value?.filter { it.contains(query, ignoreCase = true) } ?: emptyList()
        showList()
    }

    private fun handleSixDigitSearch(sixDigitCode: String) {
        viewModelScope.launch {
            val fourDigitCode = sixDigitCode.substring(0, 4)
            val targetFile = _allFiles.value?.find { it.startsWith(fourDigitCode) }

            if (targetFile != null) {
                val content = repository.readFile(targetFile)
                val spannableContent = SpannableString(content)

                // Find the exact code to highlight (e.g., 1001.00 or 100100)
                val regex = "($fourDigitCode\\.?${sixDigitCode.substring(4)})".toRegex()
                regex.findAll(content).forEach { matchResult ->
                    spannableContent.setSpan(
                        BackgroundColorSpan(0xFFFFFF00.toInt()), // Yellow highlight
                        matchResult.range.first,
                        matchResult.range.last + 1,
                        Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                    )
                }
                _fileContent.postValue(spannableContent)
                _uiState.postValue(UiState.ShowContent)
            } else {
                // File not found, show empty result in list view
                _filteredFiles.value = emptyList()
                showList()
            }
        }
    }

    fun loadFile(fileName: String) {
        viewModelScope.launch {
            val content = repository.readFile(fileName)
            _fileContent.postValue(SpannableString(content)) // No highlighting for simple click
            _uiState.postValue(UiState.ShowContent)
        }
    }

    fun showList() {
        _uiState.value = UiState.ShowList
        // When going back to list, clear the search to show all files
        if (_filteredFiles.value != _allFiles.value) {
            _filteredFiles.value = _allFiles.value
        }
    }

    sealed class UiState {
        object ShowList : UiState()
        object ShowContent : UiState()
    }

    // Factory
    class DBViewModelFactory(private val application: Application) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(DBViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST")
                return DBViewModel(application) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}
