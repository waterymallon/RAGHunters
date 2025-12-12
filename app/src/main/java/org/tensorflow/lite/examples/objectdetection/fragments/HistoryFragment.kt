package org.tensorflow.lite.examples.objectdetection.fragments

import android.app.AlertDialog
import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.navigation.fragment.findNavController
import androidx.recyclerview.widget.LinearLayoutManager
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.BarData
import com.github.mikephil.charting.data.BarDataSet
import com.github.mikephil.charting.data.BarEntry
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentHistoryBinding
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
        setupChartAppearance() // Initialize chart appearance
        observeViewModel()
        observePlotData()
    }

    private fun setupRecyclerView() {
        chatSessionAdapter = ChatSessionAdapter(
            onItemClicked = { item ->
                // Handle item click, navigate to detail fragment
                val action = HistoryFragmentDirections.actionHistoryFragmentToHistoryDetailFragment(item.session.id)
                findNavController().navigate(action)
            },
            onDeleteClicked = { item ->
                // Show confirmation dialog before deleting
                AlertDialog.Builder(requireContext())
                    .setTitle("Delete Session")
                    .setMessage("Are you sure you want to delete this session?")
                    .setPositiveButton("Delete") { _, _ ->
                        historyViewModel.deleteSession(item)
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

    private fun observePlotData() {
        historyViewModel.plotData.observe(viewLifecycleOwner) { chartData ->
            if (chartData.isEmpty()) {
                binding.tariffChart.visibility = View.GONE
                binding.noPlotDataView.visibility = View.VISIBLE
            } else {
                binding.tariffChart.visibility = View.VISIBLE
                binding.noPlotDataView.visibility = View.GONE
                setupTariffChart(chartData)
            }
        }
    }

    private fun setupChartAppearance() {
        binding.tariffChart.apply {
            setDrawGridBackground(false)
            setDrawBarShadow(false)
            setDrawValueAboveBar(true)
            getDescription().setEnabled(false)
            getLegend().setEnabled(false)
            setPinchZoom(false)
            setDoubleTapToZoomEnabled(false)

            // Y-Axis (left)
            getAxisLeft().apply {
                setDrawGridLines(false)
                setDrawLabels(true)
                setDrawAxisLine(false)
                setTextSize(10f)
                setAxisMinimum(0f)
                setGranularity(1f) // only whole numbers
            }
            getAxisRight().setEnabled(false) // Disable right Y-axis

            // X-Axis
            getXAxis().apply {
                setPosition(XAxis.XAxisPosition.BOTTOM)
                setDrawGridLines(false)
                setDrawAxisLine(true)
                setTextSize(10f)
                setGranularity(1f)
                setLabelRotationAngle(-45f) // Rotate labels for better readability
            }
            animateY(500) // Animation
        }
    }

    private fun setupTariffChart(chartData: Map<String, Int>) {
        val entries = arrayListOf<BarEntry>()
        val labels = arrayListOf<String>()

        // Sort data by count descending for better visualization
        val sortedData = chartData.toList().sortedByDescending { (_, count) -> count }

        sortedData.forEachIndexed { index, (code, count) ->
            entries.add(BarEntry(index.toFloat(), count.toFloat()))
            labels.add(code)
        }

        val dataSet = BarDataSet(entries, "Tariff Code Counts").apply {
            setColors(Color.parseColor("#42A5F5")) // A nice blue color
            setValueTextSize(10f)
            setValueTextColor(Color.BLACK)
        }

        val barData = BarData(dataSet)
        barData.setBarWidth(0.9f) // Set custom bar width

        binding.tariffChart.apply {
            setData(barData)
            getXAxis().setValueFormatter(IndexAxisValueFormatter(labels))
            getXAxis().setLabelCount(labels.size, false)
            setFitBars(true) // make the x-axis more concise
            notifyDataSetChanged()
            invalidate() // refresh
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
