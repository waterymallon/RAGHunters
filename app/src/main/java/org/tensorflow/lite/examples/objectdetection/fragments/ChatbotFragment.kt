package org.tensorflow.lite.examples.objectdetection.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.objectdetection.ApiService
import org.tensorflow.lite.examples.objectdetection.ChatResponse
import org.tensorflow.lite.examples.objectdetection.ReferenceDoc
import org.tensorflow.lite.examples.objectdetection.SharedViewModel
import org.tensorflow.lite.examples.objectdetection.TariffInfo

sealed class ChatMessage {
    data class UserQuestion(val question: String) : ChatMessage()
    data class BotResponse(val response: ChatResponse) : ChatMessage()
    data class Error(val message: String) : ChatMessage()
    object Loading : ChatMessage()
}

class ChatbotFragment : Fragment() {

    private val sharedViewModel: SharedViewModel by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        return ComposeView(requireContext()).apply {
            setContent {
                // 요청하신 테마 적용
                HSChatbotTheme {
                    // 배경색 등을 명시적으로 지정하여 테마가 잘 보이게 함
                    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                        ChatScreen(sharedViewModel = sharedViewModel)
                    }
                }
            }
        }
    }
}

@Composable
fun ChatScreen(sharedViewModel: SharedViewModel) {
    val capturedImage by sharedViewModel.capturedImage.observeAsState()
    val detectedLabels by sharedViewModel.detectedLabels.observeAsState()

    var text by remember { mutableStateOf("") }
    val coroutineScope = rememberCoroutineScope()
    var chatHistory by remember { mutableStateOf<List<ChatMessage>>(emptyList()) }

    fun handleAsk(question: String) {
        if (question.isNotBlank() && chatHistory.lastOrNull() !is ChatMessage.Loading) {
            coroutineScope.launch {
                chatHistory = chatHistory + ChatMessage.UserQuestion(question) + ChatMessage.Loading

                val result = ApiService.askQuestion(question)

                chatHistory = chatHistory.dropLast(1) // Remove Loading

                result.onSuccess { response ->
                    chatHistory = chatHistory + ChatMessage.BotResponse(response)
                }.onFailure { error ->
                    chatHistory = chatHistory + ChatMessage.Error("Error: ${error.localizedMessage}")
                }
            }
        }
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        // 1. 이미지 표시 영역
        capturedImage?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "Captured Image",
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp)
            )
            Spacer(modifier = Modifier.height(16.dp))
        }

        // 2. 감지된 라벨 버튼
        detectedLabels?.let { labels ->
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(labels) { label ->
                    Button(onClick = { handleAsk(label) }) {
                        Text(text = label)
                    }
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
        }

        // 3. 채팅 리스트
        LazyColumn(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            contentPadding = PaddingValues(bottom = 16.dp)
        ) {
            items(chatHistory) { message ->
                when (message) {
                    is ChatMessage.UserQuestion -> UserQuestionCard(message.question)
                    is ChatMessage.BotResponse -> BotResponseCards(message.response)
                    is ChatMessage.Error -> ErrorCard(message.message)
                    is ChatMessage.Loading -> Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator()
                    }
                }
            }
        }

        // 4. 입력창
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Ask anything...") },
                singleLine = true
            )
            Spacer(modifier = Modifier.width(8.dp))
            Button(
                onClick = {
                    handleAsk(text)
                    text = ""
                },
                enabled = text.isNotBlank() && chatHistory.lastOrNull() !is ChatMessage.Loading
            ) {
                Text("Send")
            }
        }
    }
}

@Composable
fun UserQuestionCard(question: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
        ) {
            Text(
                text = question,
                modifier = Modifier.padding(12.dp),
                color = MaterialTheme.colorScheme.onPrimaryContainer
            )
        }
    }
}

@Composable
fun BotResponseCards(response: ChatResponse) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(end = 40.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // Explanation
        if (!response.explanation.isNullOrBlank()) {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text("🤖 Answer", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.labelLarge)
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(response.explanation)
                }
            }
        }

        // HS Code Info
        if (!response.preliminaryHsCode.isNullOrBlank()) {
            Card {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text("🔍 Estimated HS Code", fontWeight = FontWeight.Bold)
                    Text(response.preliminaryHsCode, color = MaterialTheme.colorScheme.primary)
                }
            }
        }

        // Tariff Info
        if (!response.tariffInfo.isNullOrEmpty()) {
            TariffInfoCard(response.tariffInfo)
        }

        // Docs
        if (!response.referenceDocs.isNullOrEmpty()) {
            ReferenceDocsCard(response.referenceDocs)
        }
    }
}

@Composable
fun TariffInfoCard(tariffs: List<TariffInfo>) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer)) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text("📋 Tariff Details", fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))
            tariffs.forEach { tariff ->
                Text("• Code: ${tariff.itemNumber ?: "-"}")
                Text("• Name: ${tariff.itemName ?: "-"}")
                Text("• Rate: ${tariff.rate ?: "-"}", fontWeight = FontWeight.Bold)
                HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
            }
        }
    }
}

@Composable
fun ReferenceDocsCard(docs: List<ReferenceDoc>) {
    Card {
        Column(modifier = Modifier.padding(12.dp)) {
            Text("📚 References", fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(4.dp))
            docs.take(2).forEach { doc ->
                Text("[${doc.source}]", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
                Text(
                    doc.contentSnippet ?: "",
                    style = MaterialTheme.typography.bodySmall,
                    maxLines = 3,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(4.dp))
            }
        }
    }
}

@Composable
fun ErrorCard(message: String) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
        Text(text = message, modifier = Modifier.padding(12.dp), color = MaterialTheme.colorScheme.onErrorContainer)
    }
}