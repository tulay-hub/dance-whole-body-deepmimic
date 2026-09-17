
import mujoco
import numpy as np
import time
import torch
import onnxruntime
import onnx
import os
import xml.etree.ElementTree as ET

def parse_mujoco_joint_names(xml_path):
    """Parse joint names from a MJCF XML file.

    Args:
        xml_path: Path to the MJCF XML file.

    Returns:
        list: List of joint names in the order they appear in the XML (excluding fixed joints).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    joint_names = []
    # 遍历所有<body>，然后在每个<body>中找<joint>
    for body in root.iter('body'):
        for joint in body.iter('joint'):
            # 获取关节类型，如果没有type属性，默认为铰链关节（hinge）？在MuJoCo中，默认是铰链关节。
            # 但是固定关节（fixed）我们不要。
            joint_type = joint.get('type', 'hinge')  # 如果没有指定type，则默认为'hinge'
            if joint_type == 'fixed':
                continue
            # 获取关节名称
            joint_name = joint.get('name')
            if joint_name is not None:
                joint_names.append(joint_name)

    return joint_names

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    usd_dir_path = os.path.join(BASE_DIR, "../../../lumos_ws/lumos_rl_gym/resources")
    xml_path = os.path.join(usd_dir_path, "robots/lus2/mjcf/lus2_joint21.xml")

    mujoco_joint_index=parse_mujoco_joint_names(xml_path)
    print("Mujoco joint names:", mujoco_joint_index)

    model = onnx.load('/home/congzz/workspace/BeyondMimic/scripts/logs/rsl_rl/lus2_flat/2025-09-01_09-26-02/exported/policy.onnx')
    m = mujoco.MjModel.from_xml_path(xml_path)

    run_path = []
    joint_names = []
    joint_stiffness = []
    joint_damping = []
    default_joint_pos = []
    action_scale_seq = []
    command_names = []
    anchor_body_name = []
    body_names = []

    for prop in model.metadata_props:
        values = prop.value.split(",")
        if prop.key == "run_path":
            run_path = prop.value
            print("Run path:", run_path)
        elif prop.key == "joint_names":
            joint_names = values
            print("joint_name:",joint_names)
        elif prop.key == "joint_stiffness":
            joint_stiffness = np.array([float(x) for x in values])
            print("joint_stiffness:",joint_stiffness)
        elif prop.key == "joint_damping":
            joint_damping = np.array([float(x) for x in values])
            print("joint_damping:",joint_damping)
        elif prop.key == "default_joint_pos":
            default_joint_pos = np.array([float(x) for x in values])
            print(default_joint_pos)
        elif prop.key == "command_names":
            command_names = values
            print(command_names)
        elif prop.key == "action_scale":
            action_scale_seq = np.array([float(x) for x in values])
            print(action_scale_seq)
        elif prop.key == "anchor_body_name":
            anchor_body_name = prop.value
            print("Anchor body name:", anchor_body_name)
        elif prop.key == "body_names":
            body_names = values
            print("Body names:", body_names)
        