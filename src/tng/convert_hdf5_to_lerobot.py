
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import h5py
import numpy as np
from pathlib import Path
from typing import Union, Any, Dict
import json

JOINT_LIST_UR5 = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "hand_to_left_finger"
]

VIEWS_UR5 = [
    "camera_wrist",
    "camera_global",
]

TASK_UR5 = "Pick up the red cube and place it on the green area"

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
                "shape": (7,),
                "names": JOINT_LIST_UR5,
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": JOINT_LIST_UR5,
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


dataset_root = "/home/innovation-hacking/luebbet/dev/datasets/pick_and_place/record/2025-08-05_1635"

hdf5_file = f"{dataset_root}/all_obs.hdf5"
data_dict = hdf5_to_dict(hdf5_file)["data"]
print(data_dict)

my_dataset = create_empty_lerobot_dataset(
    dataset_path=f"{dataset_root}/lerobot",
    dataset_name="luebbet/ur5_sim_pick_and_place_v4_h264_eval",
)

for episode in data_dict.items():
    task = TASK_UR5
    if episode[0] == "_attrs":
        continue
    obs_pre = episode[1]["obs_pre"]
    obs_post = episode[1]["obs_post"]
    for i in range(1, len(obs_pre[VIEWS_UR5[0]])):
        frame = {}
        frame["observation.state"] = obs_pre["joints_pos_state"][i]
        frame["action"] = obs_post["joints_pos_action"][i]
        for view in VIEWS_UR5:
            frame[f"observation.images.{view}_view"] = np.array(
                obs_pre[view][i], dtype=np.uint8
            )
        my_dataset.add_frame(
            frame=frame,
            task=task,
        )

    my_dataset.save_episode()

write_modality_ur5(dataset_root=dataset_root)

my_dataset.push_to_hub(
    commit_message="Initial upload via script",
    run_compute_stats=False       # already done in (C)
)
