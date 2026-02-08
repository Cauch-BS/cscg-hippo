import vr2p
# load data.
path = '../set-a/Set A/Tyche-A7-SetA.zarr/Set A/Tyche-A7-SetA.zarr'
data = vr2p.ExperimentData(path)

from matplotlib import cm, colors
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os 


markers = [
           {'name':'Indicator','position': [60,100]},
           {'name':'R1','position': [130,150]},
           {'name':'R2','position': [180,200]},
           {'name': 'Teleportation','position':[1000000-1,1000000+1]}] # assign a large number to indicate teleportation regions

F_big = []
spks_big = []
day_ind = []
reward_id = []
status = []
cue_set_id = []
selected_pos_big = []



for session_id in range(len(data.vr)):
#for session_id in range(2):

    if np.isnan(data.signals.multi_session.Fns[session_id][:]).any():
        continue


    vr = data.vr[session_id]
    trial = vr.trial.copy()
    lick = vr.lick
    position = vr.path.frame.copy().reset_index()

    # merge reward_id info
    position = position.merge(trial[['set','trial_number','reward_id','status']],on='trial_number',how = 'left')
    position['speed'] = position.vr2p.rolling_speed(
            window_size = 100, ignore_threshold = 7.5)
    position.loc[position['interim_number'].notna(),'speed'] = 1000
    position_speed_filtered = position.loc[(position['speed']>=0)].copy()
    position_speed_filtered ['has_lick'] = 0
    position_speed_filtered.loc[position_speed_filtered.frame.isin(lick.frame),'has_lick']=1

    # add period info.
    selected_trials = trial.loc[trial.set.isin(trial.set.unique()),'trial_number']

    #####
    position_speed_filtered.loc[position_speed_filtered.interim_number.isin(selected_trials[:-1]), 'position'] = 1000000

    # frames in trial
    selected_position = position_speed_filtered.loc[(position_speed_filtered.trial_number.isin(selected_trials[:-1])) | (position_speed_filtered.interim_number.isin(selected_trials[:-1])),['set','frame','position','trial_number','interim_number','period_number','reward_id','status','speed','has_lick']]


    # mark position.
    selected_position['position_marker'] = 'Track'
    name = ['Near','Far']
    for rid in [1,2]:
        for marker in markers:

            selected_position.loc[(selected_position.reward_id == rid) &
                                  (selected_position.position.between(marker['position'][0],marker['position'][1])),
                                  'position_marker']=f"{marker['name']}-{name[rid-1]}"

    selected_position.loc[selected_position.interim_number.isin(selected_trials[:-1]), 'position_marker'] ='Teleportation'
    ## Make the teleporation frames after a particular trial beyond to the trial
    selected_position.loc[selected_position['interim_number'].notna(),'trial_number'] = selected_position.loc[selected_position['interim_number'].notna(),'interim_number']

    F_big.append(data.signals.multi_session.Fns[session_id][:,selected_position['frame']])
    spks_big.append(data.signals.multi_session.spks[session_id][:,selected_position['frame']])
    day_len = data.signals.multi_session.Fns[session_id][:,selected_position['frame']].shape[1]
    day_ind.append(np.ones((day_len,1))*session_id)
    reward_id.append(selected_position.reward_id)
    status.append(selected_position.status)
    cue_set_id.append(selected_position.set)
    selected_pos_big.append(selected_position)



F_big_array = np.hstack(F_big)
spks_big_array = np.hstack(spks_big)
day_ind_array = np.vstack(day_ind)
reward_id_array = np.hstack(reward_id)
status_array = np.hstack(status)
cue_set_id_array = np.hstack(cue_set_id)

np.savez_compressed(
        "dataset.npz",
        F_big_array = F_big_array,
        spks_big_array = spks_big_array,
        day_ind_array = day_ind_array,
        reward_id_array = reward_id_array,
        status_array = status_array,
        cue_set_id_array = cue_set_id_array,
)

parquet_dir = "dataset_parquet"
os.makedirs(parquet_dir, exist_ok = True)

for i, df in enumerate(selected_pos_big):
    table = pa.Table.from_pandas(df, preserve_index = True)
    pq.write_table(
            table,
            f"{parquet_dir}/selected_pos_{i:03d}.parquet",
            compression="zstd",
    )

print("Extraction Complete")
