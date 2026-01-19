Debug Plots
==================

Scan results
----------------

Gradient and loss scan (no-noise, long diff in current calc)

.. image:: debug-plots/gradient_scan.png
   :alt: Scan result
   :align: center

Gradient and loss scan (no-noise, long diff in current calc) avg

.. image:: debug-plots/gradient_scan_avg.png
   :alt: Scan result avg
   :align: center

Iteration time during the loss scan

.. image:: debug-plots/gradient_scan_time.png
   :alt: Iteration time plot
   :align: center

LUT Scan results
----------------

.. image:: scan-lut-plots/gradient_scan.png
   :alt: LUT scan result
   :align: center

.. image:: scan-lut-plots/gradient_scan_avg.png
   :alt: LUT scan result avg
   :align: center

.. image:: scan-lut-plots/gradient_scan_time.png
   :alt: LUT iteration time plot
   :align: center

NLL LUT Scan results
--------------------

.. image:: scan-nll-lut-plots/gradient_scan.png
   :alt: NLL LUT scan result
   :align: center

.. image:: scan-nll-lut-plots/gradient_scan_avg.png
   :alt: NLL LUT scan result avg
   :align: center

Fit Plots
----------------

.. image:: fit-plots/eField_fit.png
   :alt: eField fit plot
   :align: center

.. image:: fit-plots/loss_fit.png
   :alt: Loss fit plot
   :align: center


LUT Fit Plots
----------------

.. image:: fit-lut-plots/eField_fit.png
   :alt: LUT eField fit plot
   :align: center

.. image:: fit-lut-plots/loss_fit.png
   :alt: LUT Loss fit plot
   :align: center

NLL Fit Plots
----------------

.. image:: fit-nll-plots/eField_fit.png
   :alt: NLL eField fit plot
   :align: center

.. image:: fit-nll-plots/loss_fit.png
   :alt: NLL Loss fit plot
   :align: center

Simulation Consistency Plots
----------------------------

Comparison against JAX reference (Standard)

.. image:: debug-plots/diff_hist_output_0.png
   :alt: Diff Hist Output 0
   :align: center

.. image:: debug-plots/grid_comparison_output_0.png
   :alt: Grid Comparison Output 0
   :align: center

Comparison against JAX reference (Parametrized)

.. image:: debug-plots/diff_hist_output_parametrized_0.png
   :alt: Diff Hist Output Parametrized 0
   :align: center

.. image:: debug-plots/grid_comparison_output_parametrized_0.png
   :alt: Grid Comparison Output Parametrized 0
   :align: center

Aggregated Simulation Comparison
--------------------------------

LUT Mode

.. image:: debug-plots/lut_diff.png
   :alt: LUT ADC Difference Distribution
   :align: center

.. image:: debug-plots/lut_diffff.png
   :alt: LUT 2D Difference Map
   :align: center

Parametrized Mode

.. image:: debug-plots/parametrized_diff.png
   :alt: Parametrized ADC Difference Distribution
   :align: center

.. image:: debug-plots/parametrized_diffff.png
   :alt: Parametrized 2D Difference Map
   :align: center