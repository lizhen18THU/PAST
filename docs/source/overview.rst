Overview
----

.. image:: PAST_Overview.png
   :width: 700px
   :align: center

Rapid development of spatial transcriptomics (ST) have increased the demand for comprehensive methods to effectively characterize spatial domains. As a prerequisite and basis for the analysis of spatial transcriptomics, latent feature extraction largely determines the performance of downstream missions like spatial domain deciphering, trajectory inference and pseudo-time analysis. Here we propose PAST, a variational graph convolutional auto-encoder for spatial transcriptomics which integrates prior information with Bayesian neural network, captures spatial information with self-attention mechanism and enables scalable application with ripple walk sampler strategy. Through comprehensive experiments on datasets generated by different technologies, we demonstrated that PAST could effectively characterize spatial domains and facilitate various downstream analysis through integrating spatial information and reference from various sources. PAST also enable time and memory-efficient application on large datasets while preserving global spatial patterns for better performance. Importantly, PAST could also facilitate accurate annotation of spatial domains and thus provide biological insights.