package org.tensorflow.lite.examples.objectdetection.fragments

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.ComposeView
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.navigation.fragment.navArgs
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.examples.objectdetection.ChatResponse
import org.tensorflow.lite.examples.objectdetection.data.model.ChatMessageEntity
import java.io.File

class HistoryDetailFragment : Fragment() {

    private val args: HistoryDetailFragmentArgs by navArgs()
    private val historyDetailViewModel: HistoryDetailViewModel by viewModels {
        HistoryDetailViewModel.HistoryDetailViewModelFactory(requireActivity().application, args.sessionId)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        return ComposeView(requireContext()).apply {
            setContent {
                HSChatbotTheme {
                    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                        HistoryDetailScreen(viewModel = historyDetailViewModel)
                    }
                }
            }
        }
    }
}

@Composable
private fun HistoryDetailScreen(viewModel: HistoryDetailViewModel) {
    val session by viewModel.session.observeAsState()
    val messages by viewModel.messages.observeAsState(emptyList())

    // produceState will load the bitmap from the file path in a coroutine
    val capturedImage by produceState<android.graphics.Bitmap?>(initialValue = null, session?.imagePath) {
        session?.imagePath?.let { path ->
            value = withContext(Dispatchers.IO) {
                BitmapFactory.decodeFile(path)
            }
        }
    }

    // Map ChatMessageEntity to ChatMessage
    val chatHistory = messages.map { entity ->
        when (entity.messageType) {
            "USER" -> ChatMessage.UserQuestion(entity.content)
            "BOT" -> {
                val response = Gson().fromJson(entity.content, ChatResponse::class.java)
                ChatMessage.BotResponse(response)
            }
            "ERROR" -> ChatMessage.Error(entity.content)
            else -> ChatMessage.Error("Unknown message type")
        }
    }

    ChatScreen(
        capturedImage = capturedImage,
        detectedLabels = session?.detectedLabels,
        chatHistory = chatHistory,
        onAskQuestion = {}, // No action in read-only mode
        onStartNewTextSession = {}, // No action in read-only mode
        isReadOnly = true
    )
}
