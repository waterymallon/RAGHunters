package org.tensorflow.lite.examples.objectdetection.fragments

import android.app.AlertDialog
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.navigation.fragment.findNavController
import androidx.recyclerview.widget.LinearLayoutManager
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentHistoryBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class HistoryFragment : Fragment() {

    private var _binding: FragmentHistoryBinding? = null
    private val binding get() = _binding!!

    private val historyViewModel: HistoryViewModel by viewModels {
        HistoryViewModel.HistoryViewModelFactory(requireActivity().application)
    }

    private lateinit var chatSessionAdapter: ChatSessionAdapter

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentHistoryBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupRecyclerView()
        observeViewModel()
    }

    private fun setupRecyclerView() {
        chatSessionAdapter = ChatSessionAdapter(
            onItemClicked = { session ->
                // Handle item click, navigate to detail fragment
                val action = HistoryFragmentDirections.actionHistoryFragmentToHistoryDetailFragment(session.id)
                findNavController().navigate(action)
            },
            onDeleteClicked = { session ->
                // Show confirmation dialog before deleting
                android.app.AlertDialog.Builder(requireContext())
                    .setTitle("Delete Session")
                    .setMessage("Are you sure you want to delete this session?")
                    .setPositiveButton("Delete") { _, _ ->
                        historyViewModel.deleteSession(session)
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }
        )
        binding.historyRecyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = chatSessionAdapter
        }
    }

    private fun observeViewModel() {
        historyViewModel.allSessions.observe(viewLifecycleOwner) { sessions ->
            if (sessions.isNullOrEmpty()) {
                binding.historyRecyclerView.visibility = View.GONE
                binding.emptyView.visibility = View.VISIBLE
            } else {
                binding.historyRecyclerView.visibility = View.VISIBLE
                binding.emptyView.visibility = View.GONE
                chatSessionAdapter.submitList(sessions)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
