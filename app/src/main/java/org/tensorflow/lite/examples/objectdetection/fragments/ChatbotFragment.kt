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
import androidx.compose.runtime.collectAsState
import org.tensorflow.lite.examples.objectdetection.ChatResponse
import org.tensorflow.lite.examples.objectdetection.ReferenceDoc
import org.tensorflow.lite.examples.objectdetection.SharedViewModel
import org.tensorflow.lite.examples.objectdetection.TariffInfo
import org.tensorflow.lite.examples.objectdetection.fragments.ChatMessage
import org.tensorflow.lite.examples.objectdetection.fragments.ChatbotViewModel



class ChatbotFragment : Fragment() {

    private val sharedViewModel: SharedViewModel by activityViewModels()
    private val chatbotViewModel: ChatbotViewModel by activityViewModels {
        ChatbotViewModel.ChatbotViewModelFactory(requireActivity().application, sharedViewModel)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        return ComposeView(requireContext()).apply {
            setContent {
                val capturedImage by chatbotViewModel.capturedImage.collectAsState()
                val detectedLabels by chatbotViewModel.detectedLabels.collectAsState()
                val chatHistory by chatbotViewModel.chatHistory.collectAsState()

                HSChatbotTheme {
                    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                        ChatScreen(
                            capturedImage = capturedImage,
                            detectedLabels = detectedLabels,
                            chatHistory = chatHistory,
                            onAskQuestion = { question -> chatbotViewModel.askQuestion(question) },
                            isReadOnly = false
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun ChatScreen(
    capturedImage: android.graphics.Bitmap?,
    detectedLabels: List<String>?,
    chatHistory: List<ChatMessage>,
    onAskQuestion: (String) -> Unit,
    isReadOnly: Boolean
) {
    var text by remember { mutableStateOf("") }

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
                    Button(
                        onClick = { onAskQuestion(label) },
                        enabled = !isReadOnly // Disable button in read-only mode
                    ) {
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

        // 4. 입력창 (only show if not read-only)
        if (!isReadOnly) {
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
                        onAskQuestion(text)
                        text = ""
                    },
                    enabled = text.isNotBlank() && chatHistory.lastOrNull() !is ChatMessage.Loading
                ) {
                    Text("Send")
                }
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