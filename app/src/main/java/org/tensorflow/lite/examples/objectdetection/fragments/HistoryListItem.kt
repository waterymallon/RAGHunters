package org.tensorflow.lite.examples.objectdetection.fragments

import org.tensorflow.lite.examples.objectdetection.TariffInfo
import org.tensorflow.lite.examples.objectdetection.data.model.ChatSession

data class HistoryListItem(
    val session: ChatSession,
    val allTariffInfo: List<TariffInfo>,
    val tariffInfoTitle: String?
)
