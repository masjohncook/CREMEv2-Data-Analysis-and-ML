filtered_lines = []
filtered_lines_apache = []
remove_files = []
for i in range(len(input_files)):
    input_file = input_files[i]
    stage_timestamps = scenarios_timestamps[i]
    tmp_filtered_lines, tmp_filtered_lines_apache = \
        ProcessDataHelper.filter_syslog(input_file, stage_timestamps[0][0], stage_timestamps[-1][1],
                                        dls_hostname)
    filtered_lines.extend(tmp_filtered_lines)
    filtered_lines_apache.extend(tmp_filtered_lines_apache)

filtered_syslog = "filtered_dataset_generation.log"
filtered_syslog_apache = "filtered_dataset_generation_apache.log"
# write to new files
filtered_syslog_path = os.path.join(result_path, filtered_syslog)
filtered_syslog_apache_path = os.path.join(result_path, filtered_syslog_apache)
with open(filtered_syslog_path, 'w+') as fw:
    fw.write("".join(filtered_lines))
with open(filtered_syslog_apache_path, 'w+') as fw:
    fw.write("".join(filtered_lines_apache))
remove_files.append(filtered_syslog_path)
remove_files.append(filtered_syslog_apache_path)

# parse logs
# tmp_output_files is a list 2d
tmp_list_of_output_files = ProcessDataHelper.parse_syslog([filtered_syslog, filtered_syslog_apache],
                                                    input_dir=result_path, output_dir=result_path)
for tmp_output_files in tmp_list_of_output_files:
    for tmp_file in tmp_output_files:
        remove_files.append(os.path.join(result_path, tmp_file))

# merge syslog and apache log
filtered_syslog_structured = os.path.join(result_path, tmp_list_of_output_files[0][0])
filtered_syslog_structured_apache = os.path.join(result_path, tmp_list_of_output_files[1][0])

df_syslog = pd.read_csv(filtered_syslog_structured)
df_apache = pd.read_csv(filtered_syslog_structured_apache)
del df_apache['Time_Apache']
df = pd.concat([df_syslog, df_apache], ignore_index=True)

# get diff services to label data
# syslog_structured = os.path.join(result_path, "{0}_structured.csv".format(filtered_syslog))
syslog_structured = "syslog_structured.csv"

# load csv file to pandas
# df = pd.read_csv(syslog_structured)
# convert time to timestamp for filtering
df['Timestamp'] = df['Time'].apply(lambda x: parse(x).timestamp())
df = df.sort_values('Timestamp')
del df['LineId']
df = df.reset_index(drop=True)
df.index = df.index.set_names(['Index'])
tmp_output = os.path.join(result_path, syslog_structured)
df.to_csv(tmp_output, encoding='utf-8', index=True)
# remove_files.append(tmp_output)

# concatenate "Component" and "EventId" to new column and delete that column late
df["ComponentEventId"] = df["Component"].astype(str) + "-" + df["EventId"].astype(str)
# set default value for: label=0 Tactic=Normal Technique=Normal Sub-Technique=Normal
df['Label'], df['Tactic'], df['Technique'], df['SubTechnique'] = [0, 'Normal', "Normal", "Normal"]

# label
for i in range(len(scenarios_timestamps)):  # each scenario
    white_list = []
    timestamps = []
    # stage
    for j in range(len(scenarios_timestamps[i])):  # each stage
        abnormal_hostnames = scenarios_abnormal_hostnames[i][j]
        normal_hostnames = scenarios_normal_hostnames[i][j]
        abnormal_df = df[(df['HostName'].isin(abnormal_hostnames))]
        normal_df = df[(df['HostName'].isin(normal_hostnames))]
        t_start = scenarios_timestamps[i][j][0]
        timestamps.append(t_start)
        t_end = scenarios_timestamps[i][j][1]
        timestamps.append(t_end)
        tmp_white_list = ProcessDataHelper.get_all_component_event_ids(abnormal_df, normal_df, t_start, t_end)
        white_list.extend(tmp_white_list)

    labels = scenarios_labels[i]
    tactics = scenarios_tactics[i]
    techniques = scenarios_techniques[i]
    sub_techniques = scenarios_sub_techniques[i]
    # label
    ProcessDataHelper.label_filtered_syslog(df, timestamps, white_list, labels, tactics,
                                            techniques, sub_techniques)

# parsing log files for each scenario and label lifecycle
df['Label_lifecycle'] = 0
for i, file_name_scenario in enumerate(log_files):
    with open(path_labels_lifecycle, "r") as f:
        data = json.load(f)
        for j in range(len(data)):
            lifecyele_name = data[j][1]
            if lifecyele_name in file_name_scenario:
                tmp_label = data[j][0]

    stage_timestamps = scenarios_timestamps[i]
    df.loc[(df['Timestamp']>=stage_timestamps[0][0]) & (df['Timestamp']<=stage_timestamps[-1][1]), 'Label_lifecycle'] = int(tmp_label)
    
    df_parsed = df[(df['Timestamp']>=stage_timestamps[0][0]) & (df['Timestamp']<=stage_timestamps[-1][1])]
    path_scenario = os.path.join(result_path, file_name_scenario)
    df_parsed.to_csv(path_scenario, encoding='utf-8', index=False)
df.loc[df.Label == 0, 'Label_lifecycle'] = 0

del df['ComponentEventId']
tmp_output = "{0}_{1}".format("original", output_file)
full_tmp_output = os.path.join(result_path, tmp_output)
df.to_csv(full_tmp_output, encoding='utf-8', index=False)

# # remove temporary files
# for remove_file in remove_files:
#     os.system("rm {0}".format(remove_file))

output_file = ProcessDataHelper.counting_vector(result_path, tmp_output, output_file)
return output_file