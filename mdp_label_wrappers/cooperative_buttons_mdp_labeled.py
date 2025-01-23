from mdps.cooperative_buttons_mdp import CooperativeButtonsEnv
import numpy as np
from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler

class CooperativeButtonsLabeled(CooperativeButtonsEnv, MDP_Labeler):

    def get_mdp_label(self, s_next, agent_id=-1, u=-1, test = False, monolithic = False):
        """
        Return the label of the next environment state and current RM state.
        """
        row, col = self.get_state_description(s_next)

        l = []

        thresh = 0.3 #0.3

        ## append real labels
        if (row, col) == self.env_settings["yellow_button"]:
           CooperativeButtonsEnv.yellow_pressed = True
           l.append('by')
        elif (row, col) == self.env_settings["green_button"]:
            CooperativeButtonsEnv.green_pressed = True
            l.append("bg")
        elif (row, col) == self.env_settings["red_button"] and agent_id == 2:
            CooperativeButtonsEnv.a2_red_pressed = True
            l.append("a2br")
        elif (row, col) == self.env_settings["red_button"] and agent_id == 3:
            CooperativeButtonsEnv.a3_red_pressed = True
            l.append("a3br")
        elif (row, col) == self.env_settings["goal_location"]:
            l.append("g")
        elif CooperativeButtonsEnv.a2_red_pressed and not CooperativeButtonsEnv.red_pressed and agent_id == 2 and (row, col) != self.env_settings["red_button"]:
            CooperativeButtonsEnv.a2_red_pressed = False
            l.append("a2lr")
        elif CooperativeButtonsEnv.a3_red_pressed and not CooperativeButtonsEnv.red_pressed and agent_id == 3 and (row, col) != self.env_settings["red_button"]:
            CooperativeButtonsEnv.a3_red_pressed = False
            l.append("a3lr")
        if CooperativeButtonsEnv.a3_red_pressed and CooperativeButtonsEnv.a2_red_pressed:
            l.append("br")
            CooperativeButtonsEnv.red_pressed = True
        return l