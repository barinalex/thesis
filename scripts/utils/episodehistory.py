import numpy as np
from scripts.constants import DT


def init_history(state) -> dict:
    """initialize dictionary of episode data"""
    return {DT.pos: [state.get_pos()],
            DT.orn: [state.get_orn()],
            DT.vel: [state.get_vel()],
            DT.ang: [state.get_ang()],
            DT.jact: [np.array([0., 0.])]}


def update_history_by_observation(history: dict, state):
    """append new observations to corresponding keys"""
    history[DT.pos].append(state.get_pos())
    history[DT.orn].append(state.get_orn())
    history[DT.vel].append(state.get_vel())
    history[DT.ang].append(state.get_ang())


def update_history_by_action(history: dict, action: np.ndarray):
    """append new action"""
    history[DT.jact].append(action)
