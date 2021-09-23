# A machine learning approach to network protection

# Expanding Threat Landscape

The proliferation and accessibility of cyber offense toolkits and weapons has resulted in a significantly enhanced threat environment for modern network systems and administrators. Networks are increasingly vulnerable to both internal ('inside man') and external attack vectors. Modern Intrusion Detection Systems (IDSs) analyze network traffic for known fingerprints from hacking groups but these systems are in constant need of updates in order to identify malign actions over a network. The amount of network data coupled with the ever-expanding threat environment offer an opportunity to apply advanced machine learning techniques to identify emerging threats in real time allowing for rapid detection and isolation of malicious network traffic.

# Purpose

The purpose of this project is to explore the application of machine learning models for analyzing network traffic to identify potential threats. In this project I will train and compare several different machine learning models using the open source network traffic data set provided by the Canadian Institute for Cybersecurity University of New Brunswick (https://www.unb.ca/cic/datasets/ids-2018.html). These data represent realistic network traffic for a medium-sized network with several client and server nodes running predominantly Windows and Linux (Ubuntu) operating systems. The cleaned datasets consist mostly of benign network traffic with a small amount of simulated cyber-attack data with 7 different attack profiles. Thus we have a realistic, imbalanced dataset in which the majority of network traffic is considered bengin and acceptable with a small percentage of traffic that must be identified, isolated, and eliminated.

# Process

I will start by analyzing and visualizing aggregate data in order to gain insight for feature detection and feature engineering. I will used predominantly supervised learning models in the initial phases of project but will consider the implications of unsupervised learning techniques such as clustering and reinforcement learning. Once I have identified and optimized the most promising models, I will analyze the processing time and memory profiles. Since network traffic must be identified rapidly in real time I work work to translate the model to a low-level language such as C++ in order to capitalize on 'bare-metal' computation in order to reduce latency.

# Benefit/Value

The final product of this research project will help inform cyberdefense practices in the ever-changing 'cat and mouse game' of cyber warfare. The hope of this project is that it will help protect secure, private data from external and internal attacks saving companies millions of dollars in financial penalties and saving every day people from becoming victims of cyber attackers, criminals, and nation-state hackers.
