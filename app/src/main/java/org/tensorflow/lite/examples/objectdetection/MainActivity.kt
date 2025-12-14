/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection

import android.os.Build
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.examples.objectdetection.databinding.ActivityMainBinding
import com.google.android.material.tabs.TabLayout
import androidx.navigation.fragment.NavHostFragment

/**
 * Main entry point into our app. This app follows the single-activity pattern, and all
 * functionality is implemented in the form of fragments.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var activityMainBinding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        setSupportActionBar(activityMainBinding.topAppBar)

        // 1. 네비게이션 컨트롤러 가져오기
        val navHostFragment = supportFragmentManager
            .findFragmentById(R.id.fragment_container) as NavHostFragment
        val navController = navHostFragment.navController

        // 2. 탭 레이아웃 설정
        activityMainBinding.tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                when (tab?.position) {
                    0 -> {
                        supportActionBar?.title = "카메라"
                        navController.navigate(R.id.camera_fragment)
                    }
                    1 -> {
                        supportActionBar?.title = "챗봇"
                        navController.navigate(R.id.chatbot_fragment)
                    }
                    2 -> {
                        supportActionBar?.title = "History"
                        navController.navigate(R.id.history_fragment)
                    }
                    3 -> {
                        supportActionBar?.title = "DB"
                        navController.navigate(R.id.db_fragment)
                    }
                }
            }

            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })

        onBackPressedDispatcher.addCallback(this, object : androidx.activity.OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (Build.VERSION.SDK_INT == Build.VERSION_CODES.Q) {
                    // Workaround for Android Q memory leak issue in IRequestFinishCallback$Stub.
                    // (https://issuetracker.google.com/issues/139738913)
                    finishAfterTransition()
                } else {
                    if (navController.currentDestination?.id == R.id.camera_fragment) {
                        finish()
                    } else {
                        navController.popBackStack()
                    }
                }
            }
        })
    }
}
