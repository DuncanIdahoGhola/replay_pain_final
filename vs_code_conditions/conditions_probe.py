import pandas as pd
import random
import os

# --- Configuration ---
# Define the input and output directories.
LEARNING_CONDITIONS_DIR = 'conditions_learning'
PROBE_CONDITIONS_DIR = 'conditions_probe'



cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)


def generate_trials_for_sequence(seq_df):
    """
    Generates match and non-match trials for a single 3-item circular sequence.

    Args:
        seq_df (pd.DataFrame): A DataFrame with 3 rows representing one
                               circular sequence (e.g., A->B, B->C, C->A).

    Returns:
        list: A list of trial dictionaries for this sequence.
    """
    # Create a dictionary to easily find the correct next stimulus.
    # e.g., {'stimuli/zèbre.png': 'stimuli/face.png', ...}
    correct_pairs = {row['stim1_img']: row['stim2_img'] for _, row in seq_df.iterrows()}
    
    # Get the set of all 3 stimuli in this specific sequence
    sequence_stimuli = set(correct_pairs.keys())
    
    trials = []
    cue_text = "Qu'est-ce qui vient ensuite ?"

    # Iterate through each pair in the sequence (A->B, B->C, C->A)
    for target_img, correct_probe_img in correct_pairs.items():
        # 1. Generate one MATCH trial for this pair
        trials.append({
            'target_img': target_img,
            'probe_img': correct_probe_img,
            'cue_text': cue_text,
            'is_match': 'match'
        })

        # 2. Generate one NON-MATCH trial for this pair
        # The non-match is the third stimulus in the sequence that is not the
        # target or the correct probe.
        # e.g., for target A, correct is B, so non-match is C.
        non_match_options = sequence_stimuli - {target_img, correct_probe_img}
        non_match_probe_img = non_match_options.pop()

        trials.append({
            'target_img': target_img,
            'probe_img': non_match_probe_img,
            'cue_text': cue_text,
            'is_match': 'non-match'
        })
        
    # This will return 6 trials total (3 match, 3 non-match) for the sequence
    return trials


def create_probe_file_for_participant(participant_id):
    """
    Reads a learning file with two sequences and creates a corresponding
    probe file with mixed trials from both.
    """
    learning_file = os.path.join(LEARNING_CONDITIONS_DIR, f'learning_conditions_{participant_id}.csv')
    output_file = os.path.join(PROBE_CONDITIONS_DIR, f'probe_conditions_{participant_id}.csv')

    if not os.path.exists(learning_file):
        print(f"Warning: {learning_file} not found. Skipping participant {participant_id}.")
        return

    # 1. Read the learning file
    df_learn = pd.read_csv(learning_file)

    # 2. Split the data into the two separate sequences
    # The first 3 rows are the 'A,B,C' sequence
    # The next 3 rows are the '1,2,3' sequence
    df_seq1 = df_learn.iloc[:3]
    df_seq2 = df_learn.iloc[3:]

    # 3. Generate trials for each sequence independently
    trials_seq1 = generate_trials_for_sequence(df_seq1) # 6 trials
    trials_seq2 = generate_trials_for_sequence(df_seq2) # 6 trials

    # 4. Combine all trials into a single list (12 total trials)
    all_trials = trials_seq1 + trials_seq2
    
    # 5. Randomize the order of all 12 trials
    random.shuffle(all_trials)

    # 6. Create a DataFrame and save to the output CSV
    df_probe = pd.DataFrame(all_trials)
    df_probe.to_csv(output_file, index=False)
    
    print(f"Successfully created {output_file} from {learning_file}.")


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure you have run 'conditions_learning.py' first to create the input files.
    
    # Create the output directory if it doesn't exist.
    print(f"Ensuring output directory '{PROBE_CONDITIONS_DIR}/' exists...")
    os.makedirs(PROBE_CONDITIONS_DIR, exist_ok=True)
    print("...directory ready.\n")

    # Loop through all participants and generate their probe files
    # This assumes you have created 100 learning files (from 1 to 100)
    for i in range(1, 101):
        create_probe_file_for_participant(i)

    print(f"\n--- All {PROBE_CONDITIONS_DIR} files have been generated. ---")
    
    # You can inspect one of the generated files to see the output
    example_file_path = os.path.join(PROBE_CONDITIONS_DIR, 'probe_conditions_1.csv')
    print(f"\nExample output from '{example_file_path}':")
    if os.path.exists(example_file_path):
        # We display the shuffled result to show the final output
        print(pd.read_csv(example_file_path))