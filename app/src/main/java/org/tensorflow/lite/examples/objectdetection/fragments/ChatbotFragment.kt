package org.tensorflow.lite.examples.objectdetection.fragments

import android.graphics.Bitmap
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

class ChatbotFragment : Fragment() {

    private val sharedViewModel: SharedViewModel by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        return ComposeView(requireContext()).apply {
            setContent {
                HSChatbotTheme {
                    ChatScreen(sharedViewModel = sharedViewModel)
                }
            }
        }
    }
}

@Composable
fun ChatScreen(sharedViewModel: SharedViewModel) {
    val capturedImage by sharedViewModel.capturedImage.observeAsState()
    val detectedLabels by sharedViewModel.detectedLabels.observeAsState()

    var chatResponse by remember { mutableStateOf<ChatResponse?>(null) }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    val coroutineScope = rememberCoroutineScope()

    fun handleAsk(question: String) {
        if (question.isNotBlank() && !isLoading) {
            coroutineScope.launch {
                isLoading = true
                errorMessage = null
                chatResponse = null // Clear previous response
                val result = ApiService.askQuestion(question)
                result.onSuccess { response ->
                    chatResponse = response
                }.onFailure { error ->
                    errorMessage = "An error occurred: ${error.message}"
                }
                isLoading = false
            }
        }
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
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

        detectedLabels?.let { labels ->
            LazyRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(labels) { label ->
                    Button(onClick = { handleAsk(label) }) {
                        Text(text = label)
                    }
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
        }

        if (isLoading) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                errorMessage?.let {
                    item { ErrorCard(it) }
                }
                chatResponse?.let { response ->
                    if (response.explanation != null) {
                        item { InfoCard("Explanation", response.explanation) }
                    }
                    if (response.preliminaryHsCode != null) {
                        item {
                            InfoCard(
                                "Preliminary Classification",
                                "HS Code: ${response.preliminaryHsCode}\nReason: ${response.preliminaryReason ?: "N/A"}"
                            )
                        }
                    }
                    if (!response.tariffInfo.isNullOrEmpty()) {
                        item { TariffInfoCard(response.effectiveHsCode, response.tariffInfo) }
                    }
                    if (!response.referenceDocs.isNullOrEmpty()) {
                        item { ReferenceDocsCard(response.referenceDocs) }
                    }
                }
            }
        }
    }
}

@Composable
fun ErrorCard(message: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)
    ) {
        Text(
            text = message,
            modifier = Modifier.padding(16.dp),
            color = MaterialTheme.colorScheme.onErrorContainer
        )
    }
}

@Composable
fun InfoCard(title: String, content: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(text = title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = content, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
fun TariffInfoCard(effectiveCode: String?, tariffs: List<TariffInfo>) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Tariff Information (Based on HS Code: ${effectiveCode ?: "N/A"})",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            tariffs.forEachIndexed { index, tariff ->
                Column(modifier = Modifier.padding(vertical = 8.dp)) {
                    Text("Code: ${tariff.itemNumber ?: "N/A"}", fontWeight = FontWeight.SemiBold)
                    Text("Name: ${tariff.itemName ?: "N/A"}", style = MaterialTheme.typography.bodySmall)
                    Text("Rate: ${tariff.rate ?: "N/A"}", style = MaterialTheme.typography.bodySmall)
                }
                if (index < tariffs.lastIndex) {
                    HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                }
            }
        }
    }
}

@Composable
fun ReferenceDocsCard(docs: List<ReferenceDoc>) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(text = "Reference Documents", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))
            docs.forEachIndexed { index, doc ->
                Column(modifier = Modifier.padding(vertical = 8.dp)) {
                    Text("Source: ${doc.source ?: "N/A"}", fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.bodySmall)
                    Text(
                        doc.contentSnippet ?: "",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                if (index < docs.lastIndex) {
                    HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                }
            }
        }
    }
}
