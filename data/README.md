# Talmi EEG dataset

`TalmiEEG.h5` is the combined trial-level dataset used by this project. It
contains 639 complete 20-item lists from 71 participants:

- `list_type == 0`: 342 mixed lists;
- `list_type == 1`: 99 pure-negative lists;
- `list_type == 2`: 198 pure-neutral lists, including Category lists.

Category and ordinary-neutral pure lists remain distinguishable through
`cond3` (`3` and `2`, respectively).

## Rebuild

The lab archive must be mounted. The default path used by
`data_preparation.ipynb` can be overridden with `TALMI_ZARUBIN_ARCHIVE`.

Run the notebook from the repository root, or invoke the builder directly:

```bash
python -m lpp_ecmr.data_preparation \
  --mixed-behavior "$ARCHIVE/Data/Behaviour/Behaviour_csv_files/All_Included_Subjects.csv" \
  --mixed-eeg "$ARCHIVE/Final_PureList_Analysis_2026-07-15/Data/Extracted_Behavioural_and_LPP_Data/Single_Trial_Behavioural_and_EEG_Data_Z.csv" \
  --pure-behavior-dir "$ARCHIVE/Data/Behaviour/Behaviour_csv_files" \
  --pure-eeg "$ARCHIVE/Final_PureList_Analysis_2026-07-15/Data/Extracted_Behavioural_and_LPP_Data/Single_Trial_Behavioural_and_EEG_Data_Z_PureList.csv" \
  --output data/TalmiEEG.h5 \
  --force
```

The builder refuses to produce a dataset unless all structural gates pass and
the 342 mixed rows match the frozen legacy field digests exactly. The same
rules are exercised independently by `tests/test_data_contract.py`.

Raw participant files stay in the lab archive. The repository tracks only the
combined HDF5 product and the executable builder. A metadata sidecar can be
requested from the CLI for an individual rebuild, but no acceptance decision
depends on it.
