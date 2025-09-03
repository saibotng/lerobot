python convert_hdf5_to_lerobot.py \
  --dataset-root /home/innovation-hacking/luebbet/dev/datasets/pick_and_place/record/2025-09-02_2057/ \
  --dataset-name "luebbet/cal3_rand150_dim0-2_fixed_target" \
  --hdf5-files all_obs.hdf5 \
  --push-to-hub \
  --commit-message "first commit" \
  --compute-stats