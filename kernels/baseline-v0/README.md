# Kernel: from local code to Kaggle submission

The competition only accepts notebook outputs, never raw CSVs.

- `kernels/baseline-v0/notebook.ipynb`: generated (not hand-written) from the
  version folder code by inlining modules into cells with Kaggle-aware paths
  (`/kaggle/input/...` via glob, output to `/kaggle/working/submission.csv`).
- `kernels/baseline-v0/kernel-metadata.json`: CPU, offline, competition data
  attached. Pushed with `kaggle kernels push`, submitted with
  `kaggle competitions submit -k <owner/slug> -f submission.csv -v N`.
- History: v1-v3 failed on hardcoded input path; v4 listed `/kaggle/input`
  and completed; v5 runs the v1 pipeline.
