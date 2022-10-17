.. module:: past
.. automodule:: past
   :noindex:

API
====

Import PAST with::
    
    import past


Model
----
.. module::past.Model
.. currentmodule::past

.. autosummary::
    :toctree: .
    
    Model.PAST
    Model.PAST.model_train
    Model.PAST.output


Utils
----
.. module::past.Utils
.. currentmodule::past

.. autosummary::
    :toctree: .
    
    Utils.setup_seed
    Utils.integration
    Utils.geary_genes
    Utils.preprocess
    Utils.get_bulk
    Utils.visualize
    Utils.Ripplewalk_sampler
    Utils.Ripplewalk_prediction
    Utils.StDataset
    Utils.optim_parameters
    Utils.spatial_prior_graph
    Utils.load_noise
    Utils.svm_annotation
    Utils.DLPFC_split


Evaluation
----
.. module::past.Evaluation
.. currentmodule::past

.. autosummary::
    :toctree: .
    
    Evaluation.svm_cross_validation
    Evaluation.cluster_refine
    Evaluation.default_louvain
    Evaluation.default_leiden
    Evaluation.run_louvain
    Evaluation.run_leiden
    Evaluation.mclust_R
    Evaluation.cluster_metrics