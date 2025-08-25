
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import h5py
import numpy as np
from pathlib import Path
from typing import Union, Any, Dict
import json
import argparse
import sys
import os

JOINT_LIST_UR5 = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "hand_to_left_finger"
]
DELTA_JOINT_LIST_UR5 = [f"delta_{joint}" for joint in JOINT_LIST_UR5]

TCP_POSE_LIST = [
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
]

DELTA_TCP_POSE_LIST = [f"delta_{pose}" for pose in TCP_POSE_LIST]

VIEWS_UR5 = [
    "camera_wrist",
    "camera_global_front",
    "camera_global_side",
]

FPS_UR5 = 20  # Frames per second for UR5 robot data

def hdf5_to_dict(
    file_path: Union[str, Path],
    load_attrs: bool = True,
    squeeze_scalars: bool = True,
) -> Dict[str, Any]:
    """
    Recursively load an HDF5 file into a (possibly deeply‑nested) dict.
    Parameters
    ----------
    file_path : str | Path
        Path to the *.h5 / *.hdf5* file you want to read.
    load_attrs : bool, default True
        If True, each group/dataset gets an \"_attrs\" key with its HDF5
        attributes (converted to native Python scalars where possible).
    squeeze_scalars : bool, default True
        If True, zero‑dimensional datasets are converted to Python scalars
        instead of 0‑D NumPy arrays.
    Returns
    -------
    dict
        A dictionary mirroring the exact layout of the HDF5 file:
        file_dict = {
            "group_or_dataset_name": ...,
            "another_group": {
                "nested_dataset": np.ndarray,
                "_attrs": { ... }           # only if load_attrs=True
            },
            ...
        }
    """
    def _read_item(obj: h5py.Group | h5py.Dataset) -> Any:
        if isinstance(obj, h5py.Dataset):
            data = obj[()]                 # read entire dataset
            if squeeze_scalars and np.ndim(data) == 0:
                data = data.item()         # -> Python scalar
            return data
        # ---- it’s a group ----
        group_dict: Dict[str, Any] = {}
        for name, item in obj.items():
            group_dict[name] = _read_item(item)
        if load_attrs and obj.attrs:
            group_dict["_attrs"] = {
                k: (v if not isinstance(v, bytes) else v.decode("utf‑8"))
                for k, v in obj.attrs.items()
            }
        return group_dict
    with h5py.File(file_path, "r") as h5file:
        return {key: _read_item(item) for key, item in h5file.items()}


def create_empty_lerobot_dataset(
        dataset_path: str,
        dataset_name: str
        ) -> LeRobotDataset:
    features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (28,),
                "names": JOINT_LIST_UR5 + DELTA_JOINT_LIST_UR5 + TCP_POSE_LIST + DELTA_TCP_POSE_LIST,
            },
            "action": {
                "dtype": "float32",
                "shape": (28,),
                "names": JOINT_LIST_UR5 + DELTA_JOINT_LIST_UR5 + TCP_POSE_LIST + DELTA_TCP_POSE_LIST,
            },
        }
     
    for view in VIEWS_UR5:
        features[f"observation.images.{view}_view"] = {
            "dtype": "video",
            "shape": [512, 512, 3],
            "names": ["height", "width", "channel"],
            "video_info": {"video.fps": FPS_UR5, "video.codec": "h264"}
        }

    return LeRobotDataset.create(
        root=dataset_path,
        repo_id=dataset_name,
        robot_type="ur5",
        fps=FPS_UR5,
        features=features,
        image_writer_threads=8,
        image_writer_processes=0,
        video_backend="decord",
    )

def write_modality_ur5(dataset_root: str):
    modality = {
        "state": {
            "robot_arm": {
                "start": 0,
                "end": 6,
            },
            "gripper": {
                "start": 6,
                "end": 7,
            },
            "delta_robot_arm": {
                "start": 7,
                "end": 13,
            },
            "delta_gripper": {
                "start": 13,
                "end": 14,
            },
            "tcp_pose": {
                "start": 14,
                "end": 21,
            },
            "delta_tcp_pose": {
                "start": 21,
                "end": 28,
            }
        },

        "action": {
            "robot_arm": {
                "start": 0,
                "end": 6,
            },
            "gripper": {
                "start": 6,
                "end": 7,
            },
            "delta_robot_arm": {
                "start": 7,
                "end": 13,
            },
            "delta_gripper": {
                "start": 13,
                "end": 14,
            },
            "tcp_pose": {
                "start": 14,
                "end": 21,
            },
            "delta_tcp_pose": {
                "start": 21,
                "end": 28,
            }
        },

        "video": {},
        "annotation": {
            "human.task_description": {
                "original_key": "task_index"
            },
            "human.validity": {}
        }
    }

    for view in VIEWS_UR5:
        modality["video"][view] = {
            "original_key": f"observation.images.{view}_view"
        }

    with open(f"{dataset_root}/lerobot/meta/modality.json", "w") as f:
        json.dump(modality, f, indent=2)

def decode_prompts_uint8(prompt_bytes, prompt_len):
    """
    Reconstructs a list of UTF-8 strings from the stored tensors.
    
    Args:
        prompt_bytes: uint8 tensor [num_envs, max_len]
        prompt_len:   int32/int64 tensor [num_envs]
    
    Returns:
        List[str] of decoded prompts
    """
    prompts = []
    for row, n in zip(prompt_bytes, prompt_len):
        # Slice the valid part of the row and turn it into bytes
        s = bytes(row[: int(n)].tolist()).decode("utf-8", errors="replace")
        prompts.append(s)
    return prompts


def stream_to_lerobot(h5_path: str, my_dataset: LeRobotDataset, start_index: int = 1):

    print(f'Processing hdf5 file: {h5_path}')
    with h5py.File(h5_path, 'r') as f:
        demo_names = list(f['data'].keys())
        print(f'Found {len(demo_names)} demos: {demo_names}')

        for demo_name in demo_names:
            demo_group = f['data'][demo_name]
            try:
                absolute_arm_joint_states = np.array(demo_group['obs_pre']['arm_joints_pos_state'])
                absolute_gripper_joint_states = np.array(demo_group['obs_pre']['gripper_joint_pos_state'])
                absolute_tcp_pose_states = np.array(demo_group['obs_pre']['tcp_pose_state'])

                absolute_arm_joint_actions = np.array(demo_group['obs_post']['arm_joints_pos_action'])
                absolute_gripper_joint_actions = np.array(demo_group['obs_post']['gripper_joint_pos_action'])
                absolute_tcp_pose_actions = np.array(demo_group['obs_post']['tcp_pose_action'])

                images = {view: np.array(demo_group['obs_pre'][view], dtype=np.uint8) for view in VIEWS_UR5}

                prompt_bytes = demo_group["obs_pre_reset"]['prompt_bytes']
                prompt_len = demo_group["obs_pre_reset"]['prompt_len']

                task = decode_prompts_uint8(prompt_bytes, prompt_len)[0]

            except KeyError:
                print(f'Demo {demo_name} is not valid, skip it')
                return False

            assert absolute_arm_joint_actions.shape[0] == absolute_arm_joint_states.shape[0] == images[list(images.keys())[0]].shape[0], \
                f"Shape mismatch in demo '{demo_name}': " \
                f"actions {absolute_arm_joint_actions.shape[0]}, states {absolute_arm_joint_states.shape[0]}, images {images[list(images.keys())[0]].shape[0]}"
            
            delta_arm_joint_actions = absolute_arm_joint_actions - absolute_arm_joint_states
            delta_gripper_joint_actions = absolute_gripper_joint_actions - absolute_gripper_joint_states
            delta_tcp_pose_actions = absolute_tcp_pose_actions - absolute_tcp_pose_states

            delta_arm_joint_states = np.roll(delta_arm_joint_actions, shift=1, axis=0)
            delta_gripper_joint_states = np.roll(delta_gripper_joint_actions, shift=1, axis=0)
            delta_tcp_pose_states = np.roll(delta_tcp_pose_actions, shift=1, axis=0)

            all_actions = np.concatenate((
                absolute_arm_joint_actions,
                absolute_gripper_joint_actions,
                delta_arm_joint_actions,
                delta_gripper_joint_actions,
                absolute_tcp_pose_actions,
                delta_tcp_pose_actions,
            ), axis=1)

            all_states = np.concatenate((
                absolute_arm_joint_states,
                absolute_gripper_joint_states,
                delta_arm_joint_states,
                delta_gripper_joint_states,
                absolute_tcp_pose_states,
                delta_tcp_pose_states,
            ), axis=1)

            total_state_frames = absolute_arm_joint_states.shape[0]
            for i in range(start_index, total_state_frames):
                frame = {}
                for view in VIEWS_UR5:
                    frame[f"observation.images.{view}_view"] = images[view][i]
                frame["observation.state"] = all_states[i]
                frame["action"] = all_actions[i]
                my_dataset.add_frame(frame=frame, task=task)

            my_dataset.save_episode()


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 files to LeRobot dataset format")
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Root directory containing the HDF5 files to convert"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name for the LeRobot dataset (default: auto-generated from path)"
    )
    parser.add_argument(
        "--hdf5-files",
        nargs="+",
        default=["all_obs.hdf5", "all_obs_failed.hdf5"],
        help="List of HDF5 files to process (default: all_obs.hdf5 all_obs_failed.hdf5)"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to Hugging Face Hub after conversion"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Initial upload via script",
        help="Commit message for hub upload (default: 'Initial upload via script')"
    )
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Compute dataset statistics when pushing to hub"
    )
    
    args = parser.parse_args()
    
    dataset_root = args.dataset_root
    
    # Validate dataset root exists
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset root '{dataset_root}' does not exist")
        sys.exit(1)
    
    # Generate dataset name if not provided
    if args.dataset_name is None:
        dataset_name = f"luebbet/{os.path.basename(dataset_root.rstrip('/'))}"
    else:
        dataset_name = args.dataset_name
    
    print(f"Dataset root: {dataset_root}")
    print(f"Dataset name: {dataset_name}")
    
    # Build full paths to HDF5 files and check they exist
    hdf5_files = []
    for filename in args.hdf5_files:
        full_path = os.path.join(dataset_root, filename)
        if os.path.exists(full_path):
            hdf5_files.append(full_path)
            print(f"Found HDF5 file: {full_path}")
        else:
            print(f"Warning: HDF5 file not found: {full_path}")
    
    if not hdf5_files:
        print("Error: No valid HDF5 files found")
        sys.exit(1)
    
    # Create LeRobot dataset
    my_dataset = create_empty_lerobot_dataset(
        dataset_path=f"{dataset_root}/lerobot",
        dataset_name=dataset_name,
    )
    
    # Process each HDF5 file
    for hdf5_file in hdf5_files:
        print(f"\nProcessing: {hdf5_file}")
        stream_to_lerobot(hdf5_file, my_dataset)

    # Write modality configuration
    print("\nWriting modality configuration...")
    write_modality_ur5(dataset_root=dataset_root)

    # Push to hub (optional)
    if args.push_to_hub:
        print("\nPushing to hub...")
        my_dataset.push_to_hub(
            commit_message=args.commit_message,
            run_compute_stats=args.compute_stats
        )
        print("Successfully pushed to hub!")
    else:
        print("\nSkipping hub upload (use --push-to-hub to enable)")
    
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
