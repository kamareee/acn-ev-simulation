import numpy as np
from acnportal.acnsim.interface import InfrastructureInfo


def infrastructure_constraints_feasible(rates, infrastructure: InfrastructureInfo):
    phase_in_rad = np.deg2rad(infrastructure.phases)
    for j, v in enumerate(infrastructure.constraint_matrix):
        a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
        line_currents = np.linalg.norm(a @ rates, axis=0)
        if not np.all(line_currents <= infrastructure.constraint_limits[j] + 1e-7):
            return False
    return True


# This function tags a specific session with a high importance level.
# To accomplish this, it creates a new session list with the desired session tagged as high and the rest as low.
# It changes event_queue module that is located in acnsim package. (C:\Users\s3955218\Anaconda3\envs\evsim\Lib\site-packages\acnportal\acnsim\events\event_queue.py)
# In the event_queue module, it changes the EventQueue class (add_event function)
# TO DO: Modify to take multiple sessions as input and tag them as high
def tag_specific_session_for_priority(session_list, session_id: str) -> list:
    modified_session_list = []
    for session in session_list:
        if session[1].ev._session_id == session_id:
            modified_session = (session[1], "high")
            modified_session_list.append(modified_session)
        else:
            modified_session_list.append(session[1])
    return modified_session_list
