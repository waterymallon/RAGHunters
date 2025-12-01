package org.tensorflow.lite.examples.objectdetection

import android.annotation.SuppressLint
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@SuppressLint("UnsafeOptInUsageError")
@Serializable
data class ChatResponse(
    @SerialName("explanation")
    val explanation: String? = null,

    @SerialName("preliminary_hs_code")
    val preliminaryHsCode: String? = null,

    @SerialName("preliminary_reason")
    val preliminaryReason: String? = null,

    @SerialName("tariff_info")
    val tariffInfo: List<TariffInfo>? = null,

    @SerialName("effective_hs_code")
    val effectiveHsCode: String? = null,

    @SerialName("reference_docs")
    val referenceDocs: List<ReferenceDoc>? = null
)

@SuppressLint("UnsafeOptInUsageError")
@Serializable
data class TariffInfo(
    @SerialName("item_number")
    val itemNumber: String? = null,

    @SerialName("item_name")
    val itemName: String? = null,
    
    @SerialName("symbol")
    val symbol: String? = null,

    @SerialName("rate")
    val rate: String? = null,

    @SerialName("unit_tax")
    val unitTax: String? = null
)

@SuppressLint("UnsafeOptInUsageError")
@Serializable
data class ReferenceDoc(
    @SerialName("source")
    val source: String? = null,

    @SerialName("content_snippet")
    val contentSnippet: String? = null
)
