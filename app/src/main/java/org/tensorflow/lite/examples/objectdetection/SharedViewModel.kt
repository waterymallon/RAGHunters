package org.tensorflow.lite.examples.objectdetection

import android.graphics.Bitmap

import androidx.lifecycle.LiveData

import androidx.lifecycle.MutableLiveData

import androidx.lifecycle.ViewModel

import kotlinx.coroutines.flow.MutableStateFlow

import kotlinx.coroutines.flow.StateFlow

import kotlinx.coroutines.flow.asStateFlow



class SharedViewModel : ViewModel() {



    private val _capturedImage = MutableLiveData<Bitmap?>()

    val capturedImage: LiveData<Bitmap?> = _capturedImage



    private val _detectionInfo = MutableLiveData<String?>()

    val detectionInfo: LiveData<String?> = _detectionInfo



    private val _detectedLabels = MutableLiveData<List<String>?>()

    val detectedLabels: LiveData<List<String>?> = _detectedLabels



    private val _newCaptureInitiated = MutableStateFlow(false)

    val newCaptureInitiated: StateFlow<Boolean> = _newCaptureInitiated.asStateFlow()



    fun setNewCaptureInitiated(isInitiated: Boolean) {

        _newCaptureInitiated.value = isInitiated

    }



    fun setCaptureData(bitmap: Bitmap, info: String, labels: List<String>) {

        _capturedImage.value = bitmap

        _detectionInfo.value = info

        _detectedLabels.value = labels

        _newCaptureInitiated.value = true // Signal that a new capture has been initiated

    }



    fun clearData() {

        _capturedImage.value = null

        _detectionInfo.value = null

        _detectedLabels.value = null

        _newCaptureInitiated.value = false // Reset the flag

    }

}
