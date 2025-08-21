python convert_hdf5_to_lerobot.py \
  --dataset-root /home/innovation-hacking/luebbet/dev/datasets/pick_and_place/record/2025-08-21_1908/ \
  --dataset-name "luebbet/test_new_modalities" \
  --hdf5-files all_obs.hdf5 all_obs_failed.hdf5 \
  --push-to-hub \
  --commit-message "first commit" \
  --compute-stats