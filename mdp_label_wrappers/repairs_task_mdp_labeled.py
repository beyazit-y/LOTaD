from mdps.repairs_task_mdp import RepairsTaskEnv
import numpy as np
from mdp_label_wrappers.generic_mdp_labeled import MDP_Labeler

class RepairsTaskLabeled(RepairsTaskEnv, MDP_Labeler):

    def get_mdp_label(self, s_next, agent_id=-1, u=-1, test = False, monolithic = False):
        """
        Return the label of the next environment state and current RM state.
        """
        row, col = self.get_state_description(s_next)

        l = []

        thresh = 0.3 #0.3

        # ## append real labels
        if (row, col) == self.env_settings["yellow_button"]:
           l.append('y')
        elif (row, col) == self.env_settings["green_button"]:
            l.append("g")
        elif (row, col) == self.env_settings["red_button"]:
            l.append("r")
        elif (row, col) == self.env_settings["hq_location"] and agent_id == 1:
            RepairsTaskEnv.a1hq = True
            l.append("a1hq")
        elif (row, col) == self.env_settings["hq_location"] and agent_id == 2:
            RepairsTaskEnv.a2hq = True
            l.append("a2hq")
        elif (row, col) == self.env_settings["hq_location"] and agent_id == 3:
            RepairsTaskEnv.a3hq = True
            l.append("a3hq")
        elif RepairsTaskEnv.a1hq and not RepairsTaskEnv.signal and agent_id == 1 and (row, col) != self.env_settings["hq_location"]:
            RepairsTaskEnv.a1hq = False
            l.append("!a1hq")
        elif RepairsTaskEnv.a2hq and not RepairsTaskEnv.signal and agent_id == 2 and (row, col) != self.env_settings["hq_location"]:
            RepairsTaskEnv.a2hq = False
            l.append("!a2hq")
        elif RepairsTaskEnv.a3hq and not RepairsTaskEnv.signal and agent_id == 3 and (row, col) != self.env_settings["hq_location"]:
            RepairsTaskEnv.a3hq = False
            l.append("!a3hq")
        
        if (RepairsTaskEnv.a1hq and RepairsTaskEnv.a2hq) or (RepairsTaskEnv.a2hq and RepairsTaskEnv.a3hq) or (RepairsTaskEnv.a1hq and RepairsTaskEnv.a3hq):
            RepairsTaskEnv.signal = True
            l.append("sig")

        return l