# river-xgb
Modified streaming XGBoost models for the river library.

TODO:

Implementations of AXGB (Adaptive eXtreme Gradient Boosting)[^1] and BOASWIN-XGBoost (Bayesian Optimized Adaptive Sliding Window and XGBoost)[^2] ported to the River library for personal use. 

There exists no public implementation for BOASWIN-XGBoost and AXGB is implemented wiht [scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow), which is no longer supported in lieu of [River](https://github.com/online-ml/river).

[^1]: https://doi.org/10.48550/arXiv.2005.07353

[^2]: https://doi.org/10.1007/s11227-023-05729-8
