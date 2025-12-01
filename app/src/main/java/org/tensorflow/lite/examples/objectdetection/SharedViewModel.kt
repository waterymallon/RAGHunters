package org.tensorflow.lite.examples.objectdetection

import android.graphics.Bitmap
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class SharedViewModel : ViewModel() {

    private val _capturedImage = MutableLiveData<Bitmap?>()
    val capturedImage: LiveData<Bitmap?> = _capturedImage

    private val _detectionInfo = MutableLiveData<String?>()
    val detectionInfo: LiveData<String?> = _detectionInfo

    private val _detectedLabels = MutableLiveData<List<String>?>()
    val detectedLabels: LiveData<List<String>?> = _detectedLabels

    fun setCaptureData(bitmap: Bitmap, info: String, labels: List<String>) {
        _capturedImage.value = bitmap
        _detectionInfo.value = info
        _detectedLabels.value = labels
    }

    fun clearData() {
        _capturedImage.value = null
        _detectionInfo.value = null
        _detectedLabels.value = null
    }
}