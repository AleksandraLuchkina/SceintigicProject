import pandas as pd

def find_diff(row, well):
    if len(prev_depth_features[well]) == 0:
        prev_depth_features[well].append(row.values[4:])
        return
    diff = row.values[4:] - prev_depth_features[well][-1]
    prev_depth_features[well].append(row.values[4:])
    return diff
data_well = dict()
prev_depth_features = dict()
prev_class= dict()
def data_with_diff(data):
	new_data = pd.DataFrame()
	for well in set(data['Well Name']):
		prev_depth_features[well] = []
		prev_class[well] = []
		data_well[well] = data[data['Well Name'] == well]
		data_well[well] = data_well[well].sort_values(by=['Depth'])
		data_save = data_well[well].iloc[::-1]
		data_well[well]['diff_up'] = data_well[well].apply(lambda row:find_diff(row, well), axis=1)
		data_well[well] = data_well[well].dropna()
		data_well[well]['GR_diff_up'] = data_well[well].apply(lambda row: row['diff_up'][0], axis=1)
		data_well[well]['ILD_log10_diff_up'] = data_well[well].apply(lambda row: row['diff_up'][1], axis=1)
		data_well[well]['DeltaPHI_diff_up'] = data_well[well].apply(lambda row: row['diff_up'][2], axis=1)
		data_well[well]['PHIND_diff_up'] = data_well[well].apply(lambda row: row['diff_up'][3], axis=1)
		data_well[well]['PE_diff_up'] = data_well[well].apply(lambda row: row['diff_up'][4], axis=1)
		data_well[well]['NM_M_diff_up'] = data_well[well].apply(lambda row: row['diff_up'][5], axis=1)
		data_well[well]['RELPOS_diff_up'] = data_well[well].apply(lambda row: row['diff_up'][6], axis=1)
		new_data = pd.concat([new_data, data_well[well]])
		new_data = new_data.drop(['diff_up'], axis=1)
	return new_data


