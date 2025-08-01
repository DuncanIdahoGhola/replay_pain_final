import pandas as pd
import random
import os

# --- Configuration: Easy to change parameters for the experiment ---
NUM_TRIALS_PER_PARTICIPANT = 80
TRIALS_BEFORE_RULE_CHANGES_START = 20
PROBABILITY_OF_RULE_CHANGE = 0.5

# Pain level definitions and their numeric values for comparison
PAIN_LEVELS = {
    'High': 2,
    'Low': 1,
    'None': 0
}
# The rule change is a "rotation": High -> Low, Low -> None, None -> High
PAIN_RULE_CHANGE_MAP = {
    'High': 'Low',
    'Low': 'None',
    'None': 'High'
}

# --- Directory Setup ---
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

LEARNING_CONDITIONS_DIR = 'conditions_learning'
DECISION_CONDITIONS_DIR = 'conditions_decision'
os.makedirs(DECISION_CONDITIONS_DIR, exist_ok=True)


# --- Helper Function to read the sequences ---
def get_sequences_for_participant(participant_id):
    """Finds the two ordered stimulus sequences for a participant."""
    file_path = os.path.join(LEARNING_CONDITIONS_DIR, f'learning_conditions_{participant_id}.csv')
    try:
        df = pd.read_csv(file_path)
        seq1 = df['stim1_img'].iloc[:3].tolist()
        seq2 = df['stim1_img'].iloc[3:].tolist()
        return {'seq1': seq1, 'seq2': seq2}
    except FileNotFoundError:
        return None


# --- Core Logic to Generate Trials (UPDATED VERSION) ---
def generate_decision_trials_for_participant(participant_id, sequences):
    """
    Generates the full set of decision-making trials for one participant.
    """
    # Randomly assign the pain structures to the two learned sequences
    stim_sequences = [sequences['seq1'], sequences['seq2']]
    random.shuffle(stim_sequences)
    
    sequence_A = stim_sequences[0]
    sequence_B = stim_sequences[1]
    
    pain_structure_1 = ['High', 'Low', 'None']
    pain_structure_2 = ['Low', 'None', 'High']

    # Create the master map from each stimulus to its "structural" pain role
    stim_to_pain_role_map = {
        **dict(zip(sequence_A, pain_structure_1)),
        **dict(zip(sequence_B, pain_structure_2))
    }

    trials_list = []

    for trial_num in range(1, NUM_TRIALS_PER_PARTICIPANT + 1):
        
        # Determine if a rule change occurs
        if trial_num <= TRIALS_BEFORE_RULE_CHANGES_START:
            rule_change_info = 'no'
        else:
            rule_change_info = 'yes' if random.random() < PROBABILITY_OF_RULE_CHANGE else 'no'

        # --- CHANGE 1: Use a numeric starting position (1, 2, or 3) ---
        # The internal index will be 0, 1, or 2
        start_position_index = random.choice([0, 1, 2])

        # 5. Determine the two possible paths based on the numeric starting position.
        # Path 1 is the 2-step journey starting from the given position in Sequence A.
        # The modulo operator (%) ensures the sequence wraps around (e.g., from item 3 back to 1).
        path1_stimuli = [sequence_A[i % 3] for i in range(start_position_index, start_position_index + 3)]

        # Path 2 is the 2-step journey from the *same position* in Sequence B.
        path2_stimuli = [sequence_B[i % 3] for i in range(start_position_index, start_position_index + 3)]

        # --- CHANGE 2: Extract individual stimuli for clear columns ---
        # Path 1 stimuli
        path1_start = path1_stimuli[0]
        path1_middle = path1_stimuli[1]
        path1_end = path1_stimuli[2] # This is the final stimulus that determines pain

        # Path 2 stimuli
        path2_start = path2_stimuli[0]
        path2_middle = path2_stimuli[1]
        path2_end = path2_stimuli[2] # This is the final stimulus that determines pain

        # 6. Calculate the pain outcome for each path based on the rule
        role_path1 = stim_to_pain_role_map[path1_end]
        role_path2 = stim_to_pain_role_map[path2_end]

        if rule_change_info == 'no':
            final_pain_path1 = role_path1
            final_pain_path2 = role_path2
        else: # rule_change_info == 'yes'
            final_pain_path1 = PAIN_RULE_CHANGE_MAP[role_path1]
            final_pain_path2 = PAIN_RULE_CHANGE_MAP[role_path2]

        # 7. Determine the optimal choice
        optimal_choice = 1 if PAIN_LEVELS[final_pain_path1] < PAIN_LEVELS[final_pain_path2] else 2
        if PAIN_LEVELS[final_pain_path1] == PAIN_LEVELS[final_pain_path2]:
            optimal_choice = random.choice([1, 2])

        # 8. Store all trial information in a dictionary with the new, clear column names
        trials_list.append({
            'participant_id': participant_id,
            'trial_num': trial_num,
            'start_position': start_position_index + 1, # Save as 1, 2, or 3 for clarity
            'rule_change_info': rule_change_info,
            'path1_start_stim': path1_start,
            'path1_middle_stim': path1_middle,
            'path1_end_stim': path1_end,
            'path2_start_stim': path2_start,
            'path2_middle_stim': path2_middle,
            'path2_end_stim': path2_end,
            'path1_final_pain': final_pain_path1,
            'path2_final_pain': final_pain_path2,
            'optimal_choice': f'path_{optimal_choice}'
        })

    return pd.DataFrame(trials_list)


# --- Main Execution Block (unchanged) ---
if __name__ == "__main__":
    print(f"--- Generating decision-making condition files for participants 1-99 ---")

    for i in range(1, 100):
        participant_sequences = get_sequences_for_participant(i)
        
        if not participant_sequences:
            print(f"Warning: Could not find learning file for participant {i}. Skipping.")
            continue
            
        df_decision = generate_decision_trials_for_participant(i, participant_sequences)
        
        output_filename = os.path.join(DECISION_CONDITIONS_DIR, f'decision_conditions_{i}.csv')
        df_decision.to_csv(output_filename, index=False)
        
    print(f"\n--- Process complete. ---")
    print(f"All decision condition files saved in '{DECISION_CONDITIONS_DIR}/' directory.")

    example_file = os.path.join(DECISION_CONDITIONS_DIR, 'decision_conditions_1.csv')
    if os.path.exists(example_file):
        print(f"\n--- Example: First 5 trials for participant 1 (new format) ---")
        pd.set_option('display.max_columns', None) # Show all columns
        print(pd.read_csv(example_file).head())