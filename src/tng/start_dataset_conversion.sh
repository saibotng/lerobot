python convert_hdf5_to_lerobot.py \
  --dataset-root /home/innovation-hacking/luebbet/dev/datasets/pick_and_place/record/2025-09-04_2157/ \
  --dataset-name "luebbet/validation_dataset" \
  --hdf5-files all_obs.hdf5 \
  --push-to-hub \
  --commit-message "first commit" \
  --compute-stats