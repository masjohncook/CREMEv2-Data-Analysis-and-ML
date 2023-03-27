import pandas as pd
import sys
import json
from nfstream import NFStreamer
import os


root_folder = os.path.join("/", "RD", "CREMEv2_Result", "20230207", "logs")
scenario = ["mirai", "disk_wipe", "ransomware", "resource_hijacking", "end_point_dos"]
scenario_folder = ["01_mirai", "02_disk_wipe", "03_ransomware", "04_resource_hijacking", "05_end_point_dos"]
traffic_folder = "traffic"
label_folder = "label_traffic"
filename = "traffic.pcap"
label = {'mirai': [1, 2, 3, 5, 9, 11, 12, 13],
         'disk_wipe': [1, 2, 4, 6, 8, 14],
         'ransomware': [1, 2, 4, 6, 7, 12, 15],
         'resource_hijacking': [1, 2, 4, 6, 8, 12, 16],
         'end_point_dos': [1, 2, 4, 8, 10, 12, 17]
         }



def feature_extraction():
    try:
        for n in range(len(scenario)):
            path = os.path.join(root_folder, scenario_folder[n], traffic_folder, filename)
            final_filename = "label_traffic_{}_nfstream.csv".format(scenario[n])
            final_filepath = os.path.join(root_folder, label_folder, "nfstream", final_filename)
            print("Processing the "+scenario[n]+" on "+path + " to " + final_filepath)
            flows_count = NFStreamer(source=path).to_csv(path=final_filepath,
                                                            columns_to_anonymize=(),
                                                            flows_per_file=0,
                                                            rotate_files=0)
            print("==================================================================")
            print(flows_count)

    except Exception as e:
        print(e)


def get_timestamps_mirai(scenario, log_folder):
    label_number = 0
    timestamp_namelist = []
    timestamps = []
    timestamp_num = label_num * 2
    
    try:
        # step 1
        timestamp_namelist.append(os.path.join(log_folder, "time_step_1_mirai_start.txt"))
        timestamp_namelist.append(os.path.join(log_folder, "time_step_1_mirai_end.txt"))
        # step 2
        timestamp_namelist.append(os.path.join(log_folder, "time_step_2_mirai_start.txt"))
        timestamp_namelist.append(os.path.join(log_folder, "time_step_2_mirai_end.txt"))
        # step 3
        timestamp_namelist.append(os.path.join(log_folder, "time_step_3_mirai_start_cnc_and_login.txt"))
        timestamp_namelist.append(os.path.join(log_folder, "time_step_4_start_DDoS.txt"))
        # step 4
        timestamp_namelist.append(os.path.join(log_folder, "time_step_4_start_DDoS.txt"))
        timestamp_namelist.append(os.path.join(log_folder, "time_step_5_kali_start_scan.txt"))
        # step 5
        timestamp_namelist.append(os.path.join(log_folder, "time_step_5_kali_start_scan.txt"))
        timestamp_namelist.append(os.path.join(log_folder, "time_step_6_mirai_wait_finish_scan.txt"))
        # step 6
        timestamp_namelist.append(os.path.join(log_folder, "time_step_6_mirai_wait_finish_scan.txt"))
        timestamp_namelist.append(os.path.join(log_folder, "time_step_6_MaliciousClient_stop_malicious.txt"))
        # step 7
        timestamp_namelist.append(os.path.join(log_folder, "time_step_7_start_transfer.txt"))
        timestamp_namelist.append(os.path.join(log_folder, "time_step_7_mirai_wait_finish_transfer.txt"))
        # step 8
        timestamp_namelist.append(os.path.join(log_folder, "time_step_7_mirai_wait_finish_transfer.txt"))
        timestamp_namelist.append(os.path.join(log_folder, "time_step_8_mirai_wait_finish_ddos.txt"))

        for i in range(timestamp_num):
            with open(timestamp_namelist[i], 'rt') as f:
                # read to sec
                timestamps.append(f.readline(10))
        timestamps = [int(float(i)) for i in timestamps]
        # In original CREME, ddos duration time was added
        timestamps[timestamp_num-1] += 10
        # timestamps[timestamp_num-1] += (10 + int(dur)) # 10 to avoid problems if there is some delay
        return timestamps
        

    except Exception as e:
        print(e)
        
        
def get_timestamps_non_mirai(scenario, log_folder):
    label_number = 0
    timestamp_namelist = []
    timestamps = []
    timestamp_num = label_number * 2
    
    try:
        # get start and end timestamps
        for i in range(label_number):
            i += 1
            timestamp_namelist.append(os.path.join(log_folder, "time_step_"+str(i)+"_start.txt"))
            timestamp_namelist.append(os.path.join(log_folder, "time_step_"+str(i)+"_end.txt"))

        for i in range(timestamp_num):
            with open(timestamp_namelist[i], 'rt') as f:
                # read to sec
                timestamps.append(f.readline(10))

        timestamps = [int(float(i)) for i in timestamps]
        return timestamps

    except Exception as e:
        print(e)
        
def process_data():
    try:
        for data in scenario_folder():
    
    except Exception as e:
        print(e)
        

