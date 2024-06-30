# Soc/Soh Estimation Framework Research Replication
## Code written by Ryley Traverse

[Link to Modeling Notebook](https://github.com/ryleytraverse/ssef_research_replication/blob/main/soc_soh_estimation_framework_modeling.ipynb)

This project focuses on replicating the research found in [this paper](https://www.sciencedirect.com/science/article/abs/pii/S036054422303195X, 'A novel integrated SOC-SOH estimation framework for whole-life-cycle lithium-ion batteries'). Estimation of the current charge and health of a battery during cycling (charging/discharging) is critical for battery management systems (BMS) to accurately and safely operate lithium-ion batteries. Additionally, precise SOH estimation influences the accuracy of SOC estimations and contributes to preventing cell events due to untimely battery replacement when high capacity degradation is present. SOC and SOH must be estimated due to challenges in capturing the true capacity of a battery during everyday charging and discharging. There are currently two main non-model-based approaches to SOC estimation: Open Circuit Voltage (OCV) and Coulomb counting. Coulomb counting has compounding issues due to its cumulative approach and OCV doesn't make sense for real-world applications due to the necessity of load disconnection.

The approach suggested in this paper is superior to current model-based solutions for a multitude of reasons. The first reason is due to the obvious relationship between voltage, current, SOC, and SOH. Current solutions rely on separate models for estimation without considering the increase in performance you can obtain by using a single joint model. Not only can you see an increase in performance of both SOC and SOH estimation with a single model, but, due to parameter sharing and segmented training of the model, the benefits extend to model size, complexity, and efficiency.

It uses an open-source dataset available from the Center of Advanced Life Cycle Engineering at the University of Maryland
[Link to Dataset](https://calce.umd.edu/battery-data#CS2, 'CALCE Data')

The data processing steps are laid out [here](https://github.com/ryleytraverse/ssef_research_replication/blob/main/soc_soh_estimation_framework_data_prep.ipynb)
