""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os

from gym.envs.registration import register
from mj_envs.envs.relay_kitchen.kitchen_multitask_v1 import KitchenTasksV0
from mj_envs.envs.relay_kitchen.franka_kitchen_v1 import KitchenFrankaFixed, \
    KitchenFrankaRandom, KitchenFrankaDemo

print("RS:> Registering Kitchen Envs")

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = CURR_DIR + "/assets/franka_kitchen.xml"
CONFIG_PATH = CURR_DIR + "/assets/franka_kitchen.config"

# Kitchen
register(
    id="kitchen-v0",
    entry_point="mj_envs.envs.relay_kitchen:KitchenTasksV0",
    max_episode_steps=280,
)

DEMO_ENTRY_POINT = "mj_envs.envs.relay_kitchen:KitchenFrankaDemo"
RANDOM_ENTRY_POINT = "mj_envs.envs.relay_kitchen:KitchenFrankaRandom"
FIXED_ENTRY_POINT = "mj_envs.envs.relay_kitchen:KitchenFrankaFixed"
ENTRY_POINT = RANDOM_ENTRY_POINT

# ========================================================

# V3 environments
# In this version of the environment, the observations consist of the
# distance between end effector and all relevent objects in the scene

state_obs_keys_wt = {"robot_jnt": 1.0, "objs_jnt": 1.0, "obj_goal": 1.0,
                     "end_effector": 1.0}
visual_obs_keys_wt = {"robot_jnt": 1.0,
                      "end_effector": 1.0,
                      # "rgb:right_cam:224x224:r3m18": 1.0,
                      # "rgb:left_cam:224x224:r3m18": 1.0,
                      # "rgb:right_cam:224x224:flat": 1.0,
                      # "rgb:left_cam:224x224:flat": 1.0,
                      }
obs_keys_wt = visual_obs_keys_wt
for site in KitchenFrankaFixed.OBJ_INTERACTION_SITES:
    obs_keys_wt[site + "_err"] = 1.0

# Kitchen
register(
    id="visual_kitchen-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=280,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {},
        "obj_init": {
            "knob1_joint": 0,
            "knob2_joint": 0,
            "knob3_joint": 0,
            "knob4_joint": 0,
            "lightswitch_joint": 0,
            "slidedoor_joint": 0,
            "micro0joint": 0,
            "rightdoorhinge": 0,
            "leftdoorhinge": 0,
        },
        "obs_keys_wt": obs_keys_wt,
    },
)

# Kitchen
register(
    id="visual_kitchen_close-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_goal": {},
        "obj_init": {
            "knob1_joint": -1.57,
            "knob2_joint": -1.57,
            "knob3_joint": -1.57,
            "knob4_joint": -1.57,
            "lightswitch_joint": -0.7,
            "slidedoor_joint": 0.44,
            "micro0joint": -1.25,
            "rightdoorhinge": 1.57,
            "leftdoorhinge": -1.25,
        },
        "obs_keys_wt": obs_keys_wt,
    },
)

# Microwave door
register(
    id="visual_kitchen_micro_open-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"micro0joint": 0},
        "obj_goal": {"micro0joint": -1.25},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "microhandle_site",
    },
)
register(
    id="visual_kitchen_micro_close-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"micro0joint": -1.25},
        "obj_goal": {"micro0joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "microhandle_site",
    },
)

# Right hinge cabinet
register(
    id="visual_kitchen_rdoor_open-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"rightdoorhinge": 0},
        "obj_goal": {"rightdoorhinge": 1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "rightdoor_site",
    },
)
register(
    id="visual_kitchen_rdoor_close-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"rightdoorhinge": 1.57},
        "obj_goal": {"rightdoorhinge": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "rightdoor_site",
    },
)

# Left hinge cabinet
register(
    id="visual_kitchen_ldoor_open-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"leftdoorhinge": 0},
        "obj_goal": {"leftdoorhinge": -1.25},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "leftdoor_site",
    },
)
register(
    id="visual_kitchen_ldoor_close-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"leftdoorhinge": -1.25},
        "obj_goal": {"leftdoorhinge": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "leftdoor_site",
    },
)

# Slide cabinet
register(
    id="visual_kitchen_sdoor_open-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"slidedoor_joint": 0},
        "obj_goal": {"slidedoor_joint": 0.44},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "slide_site",
    },
)
register(
    id="visual_kitchen_sdoor_close-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"slidedoor_joint": 0.44},
        "obj_goal": {"slidedoor_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "slide_site",
    },
)

# Lights
register(
    id="visual_kitchen_light_on-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"lightswitch_joint": 0},
        "obj_goal": {"lightswitch_joint": -0.7},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "light_site",
    },
)
register(
    id="visual_kitchen_light_off-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"lightswitch_joint": -0.7},
        "obj_goal": {"lightswitch_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "light_site",
    },
)

# Knob4
register(
    id="visual_kitchen_knob4_on-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob4_joint": 0},
        "obj_goal": {"knob4_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob4_site",
    },
)
register(
    id="visual_kitchen_knob4_off-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob4_joint": -1.57},
        "obj_goal": {"knob4_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob4_site",
    },
)

# Knob3
register(
    id="visual_kitchen_knob3_on-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob3_joint": 0},
        "obj_goal": {"knob3_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob3_site",
    },
)
register(
    id="visual_kitchen_knob3_off-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob3_joint": -1.57},
        "obj_goal": {"knob3_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob3_site",
    },
)

# Knob2
register(
    id="visual_kitchen_knob2_on-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob2_joint": 0},
        "obj_goal": {"knob2_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob2_site",
    },
)
register(
    id="visual_kitchen_knob2_off-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob2_joint": -1.57},
        "obj_goal": {"knob2_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob2_site",
    },
)

# Knob1
register(
    id="visual_kitchen_knob1_on-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob1_joint": 0},
        "obj_goal": {"knob1_joint": -1.57},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob1_site",
    },
)
register(
    id="visual_kitchen_knob1_off-v3",
    entry_point=ENTRY_POINT,
    max_episode_steps=50,
    kwargs={
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "obj_init": {"knob1_joint": -1.57},
        "obj_goal": {"knob1_joint": 0},
        "obs_keys_wt": obs_keys_wt,
        "interact_site": "knob1_site",
    },
)

# Franka Appliance ======================================================================
from mj_envs.envs.relay_kitchen.franka_appliance_v1 import FrankaAppliance

# MICROWAVE
obs_keys_wt = {
    "robot_jnt": 1.0,
    # "objs_jnt": 1.0,
    # "obj_goal": 1.0,
    "end_effector": 1.0,
    # "microhandle_site_err": 1
}
register(
    id="fvisual_ranka_micro_open-v3",
    entry_point="mj_envs.envs.relay_kitchen:FrankaAppliance",
    max_episode_steps=75,
    kwargs={
        "model_path": CURR_DIR + "/assets/franka_microwave.xml",
        "config_path": CURR_DIR + "/assets/franka_microwave.config",
        "obj_init": {"micro0joint": 0},
        "obj_goal": {"micro0joint": -1.25},
        "obj_interaction_sites": ("microhandle_site",),
        "obj_jnt_names": ("micro0joint",),
        "interact_site": "microhandle_site",
    },
)
register(
    id="visual_franka_micro_close-v3",
    entry_point="mj_envs.envs.relay_kitchen:FrankaAppliance",
    max_episode_steps=50,
    kwargs={
        "model_path": CURR_DIR + "/assets/franka_microwave.xml",
        "config_path": CURR_DIR + "/assets/franka_microwave.config",
        "obj_init": {"micro0joint": -1.25},
        "obj_goal": {"micro0joint": 0},
        "obj_interaction_sites": ("microhandle_site",),
        "obj_jnt_names": ("micro0joint",),
        "interact_site": "microhandle_site",
    },
)
register(
    id="visual_franka_micro_random-v3",
    entry_point="mj_envs.envs.relay_kitchen:FrankaAppliance",
    max_episode_steps=50,
    kwargs={
        "model_path": CURR_DIR + "/assets/franka_microwave.xml",
        "config_path": CURR_DIR + "/assets/franka_microwave.config",
        "obj_init": {"micro0joint": (-1.25, 0)},
        "obj_goal": {"micro0joint": (-1.25, 0)},
        "obj_interaction_sites": ("microhandle_site",),
        "obj_jnt_names": ("micro0joint",),
        "obj_body_randomize": ("microwave",),
        "interact_site": "microhandle_site",
    },
)

# SLIDE-CABINET
obs_keys_wt = {
    "robot_jnt": 1.0,
    # "objs_jnt": 1.0,
    # "obj_goal": 1.0,
    "end_effector": 1.0,
    # "slide_site_err": 1
}
register(
    id="visual_franka_slide_open-v3",
    entry_point="mj_envs.envs.relay_kitchen:FrankaAppliance",
    max_episode_steps=50,
    kwargs={
        "model_path": CURR_DIR + "/assets/franka_slidecabinet.xml",
        "config_path": CURR_DIR + "/assets/franka_slidecabinet.config",
        "obj_init": {"slidedoor_joint": 0},
        "obj_goal": {"slidedoor_joint": .44},
        "obj_interaction_sites": ("slide_site",),
        "obj_jnt_names": ("slidedoor_joint",),
        "interact_site": "slide_site",
    },
)
register(
    id="visual_franka_slide_close-v3",
    entry_point="mj_envs.envs.relay_kitchen:FrankaAppliance",
    max_episode_steps=50,
    kwargs={
        "model_path": CURR_DIR + "/assets/franka_slidecabinet.xml",
        "config_path": CURR_DIR + "/assets/franka_slidecabinet.config",
        "obj_init": {"slidedoor_joint": .44},
        "obj_goal": {"slidedoor_joint": 0},
        "obj_interaction_sites": ("slide_site",),
        "obj_jnt_names": ("slidedoor_joint",),
        "interact_site": "slide_site",
    },
)
register(
    id="visual_franka_slide_random-v3",
    entry_point="mj_envs.envs.relay_kitchen:FrankaAppliance",
    max_episode_steps=50,
    kwargs={
        "model_path": CURR_DIR + "/assets/franka_slidecabinet.xml",
        "config_path": CURR_DIR + "/assets/franka_slidecabinet.config",
        "obj_init": {"slidedoor_joint": (0, .44)},
        "obj_goal": {"slidedoor_joint": (0, .44)},
        "obj_interaction_sites": ("slide_site",),
        "obj_jnt_names": ("slidedoor_joint",),
        "obj_body_randomize": ("slidecabinet",),
        "interact_site": "slide_site",
    },
)
