import os
import json
import numpy as np
from argparse import ArgumentParser

from arguments import ModelParams
from scene import Scene, GaussianModel


def _COLMAP_to_OpenGL(colmap):
    colmap = colmap.cpu().numpy()

    opengl = np.linalg.inv(colmap).T
    opengl[:3, 1:3] *= -1
    return opengl

def binary_to_transform_json(model_path: str, data_path: str):
    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args(["-s", data_path, "-m", model_path])
    dataset = lp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    cameras = sum(scene.train_cameras.values(), [])
    assert len(cameras) > 0, "Cameras not found"

    JSON = {
        "camera_angle_x": cameras[0].FoVx, # Same as blender dataset
        "frames": [
            {
                "file_path": f"./train/{camera.image_name}",
                "rotation": 0.012566370614359171, # Unused,
                "transform_matrix": _COLMAP_to_OpenGL(camera.world_view_transform).tolist(),
            }
            for camera in cameras
        ]
    }

    with open(os.path.join(data_path, "transforms_train.json"), "w") as f:
        json.dump(JSON, f)

    # Create empty test and val transforms
    with open(os.path.join(data_path, "transforms_test.json"), "w") as f:
        json.dump({"camera_angle_x": 0, "frames" : []}, f)

    with open(os.path.join(data_path, "transforms_val.json"), "w") as f:
        json.dump({"camera_angle_x": 0, "frames" : []}, f)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-s", "--data_path", type=str, required=True)
    args = parser.parse_args()

    binary_to_transform_json(args.model_path, args.data_path)