package org.tensorflow.lite.examples.objectdetection.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SearchView
import androidx.activity.OnBackPressedCallback
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentDbBinding

class DBFragment : Fragment() {

    private var _binding: FragmentDbBinding? = null
    private val binding get() = _binding!!

    private val viewModel: DBViewModel by viewModels {
        DBViewModel.DBViewModelFactory(requireActivity().application)
    }

    private lateinit var fileListAdapter: FileListAdapter

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentDbBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupRecyclerView()
        setupSearchView()
        observeViewModel()
        handleBackButton()
    }

    private fun setupRecyclerView() {
        fileListAdapter = FileListAdapter { fileName ->
            viewModel.loadFile(fileName)
        }
        binding.docListRecyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = fileListAdapter
        }
    }

    private fun setupSearchView() {
        binding.docSearchView.setOnQueryTextListener(object : SearchView.OnQueryTextListener {
            override fun onQueryTextSubmit(query: String?): Boolean {
                viewModel.performSearch(query)
                return true
            }

            override fun onQueryTextChange(newText: String?): Boolean {
                // You can have live search here if desired
                viewModel.performSearch(newText)
                return true
            }
        })
    }

    private fun observeViewModel() {
        viewModel.displayedFiles.observe(viewLifecycleOwner) { files ->
            fileListAdapter.submitList(files)
        }

        viewModel.fileContent.observe(viewLifecycleOwner) { content ->
            binding.docContentTextView.text = content
        }

        viewModel.uiState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is DBViewModel.UiState.ShowList -> {
                    binding.docListRecyclerView.visibility = View.VISIBLE
                    binding.docContentScrollView.visibility = View.GONE
                }
                is DBViewModel.UiState.ShowContent -> {
                    binding.docListRecyclerView.visibility = View.GONE
                    binding.docContentScrollView.visibility = View.VISIBLE
                }
            }
        }
    }

    private fun handleBackButton() {
        val callback = object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (viewModel.uiState.value is DBViewModel.UiState.ShowContent) {
                    viewModel.showList()
                } else {
                    // If we are already in the list view, let the default back action happen
                    isEnabled = false
                    requireActivity().onBackPressedDispatcher.onBackPressed()
                }
            }
        }
        requireActivity().onBackPressedDispatcher.addCallback(viewLifecycleOwner, callback)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
