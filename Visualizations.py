# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 06:13:33 2022

@author: 12105
"""

# Visualize/learn data:
tot_fwd_pkts = sns.swarmplot(data = X_res, y = "Tot Fwd Pkts", x = "Label", hue='Label')

tot_bwd_pkts = sns.swarmplot(data = X_res, y = "Tot Bwd Pkts", x = "Label")
flow_duration = sns.swarmplot(data = X_res, y = "Flow Duration", x = "Label")

#sns.swarmplot(data = X_res, y = "Flow Duration", x = "Protocol")

sns.swarmplot(data = X_res, y = "Tot Fwd Pkts", x = "Label_Bin")

sns.swarmplot(data = X_res, y = "Tot Bwd Pkts", x = "Label_Bin")
sns.swarmplot(data = X_res, y = "Flow Duration", x = "Label_Bin")

sns.swarmplot(data = X_res, y = "Flow Duration", x = "Label_Bin")

#Time series Tot FWd Pkts by category
sns.lineplot(x="Timestamp", y = "Tot Fwd Pkts", hue='Label', data = X_res)
#sns.lineplot(x = X_train["Timestamp"], y = "Pkt Len Mean", hue="Label", data = X_train)

X_res['Label'] = y_res
# violin plot to look at Fwd Packets
Total_Fwd_Pckts_plt = sns.violinplot(x="Tot Fwd Pkts", y="Label", data=X_res)
Total_Bwd_Pckts_plt = sns.violinplot(x="Tot Bwd Pkts", y="Label", data=X_res)
