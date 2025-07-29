import pandas as pd
import random
import os



#Lets start with some conditions that will be usefull later on

NUM_FILES_TO_GENERATE = 100
IMAGES = ['face', 'zèbre', 'banane', 'ciseau','violon','lunette']
TRIALS_PER_IMAGE_TYPE = 36 
MAX_CONSECUTIVE_IDENTICAL_IMAGES = 2
MAX_CONSECUTIVE_MATCH_MISMATCH = 2 

#find our current directory
cwd = os.getcwd()
#exit one level + make sure the outpout_dir exists
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)
OUTPUT_DIR = 'func_loc_conditions'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 1. Generate Base Trials (Once) ---
def generate_base_trials(images, trials_per_type):
    base_rows = []
    for img in images:
        # MATCH trials
        for _ in range(trials_per_type):
            base_rows.append({
                'image_file': f'stimuli/{img}.png',
                'presented_word': img,
                'is_match': 'match',
                'blank_duration': round(random.uniform(1.0, 2.0), 2), # Will be re-randomized per file
                'iti_duration': round(random.uniform(1.0, 3.0), 2)    # Will be re-randomized per file
            })

        # MISMATCH trials
        mismatches = [w for w in images if w != img]
        for _ in range(trials_per_type):
            word = random.choice(mismatches)
            base_rows.append({
                'image_file': f'stimuli/{img}.png',
                'presented_word': word,
                'is_match': 'mismatch',
                'blank_duration': round(random.uniform(1.0, 2.0), 2), # Will be re-randomized
                'iti_duration': round(random.uniform(1.0, 3.0), 2)    # Will be re-randomized
            })
    return base_rows

# --- Helper for shuffle_with_constraint ---
def check_local_validity(trials, check_idx, key_column, max_allowed_consecutive):
    """
    Checks if the item at trials[check_idx] is part of any violating sequence
    for the given key_column and max_allowed_consecutive.
    A sequence of (max_allowed_consecutive + 1) identical items is a violation.
    Returns True if valid (no violation involving check_idx), False otherwise.
    """
    n = len(trials)
    if n == 0: return True

    # Iterate over all possible start positions 's' of a sequence of length (max_allowed_consecutive + 1)
    # such that trials[check_idx] is part of that sequence.
    # The sequence is trials[s], ..., trials[s + max_allowed_consecutive].
    # Loop for s: from (check_idx - max_allowed_consecutive) TO check_idx.
    # Clamped to valid list indices for the start of the sequence.
    # Max start index for a sequence of L items is n - L. L = max_allowed_consecutive + 1.
    # So, max start index is n - (max_allowed_consecutive + 1).
    
    start_s_loop = max(0, check_idx - max_allowed_consecutive)
    end_s_loop = min(check_idx, n - (max_allowed_consecutive + 1))

    for s in range(start_s_loop, end_s_loop + 1):
        # Sequence under test: trials[s]...trials[s+max_allowed_consecutive]
        is_potential_violation = True
        first_val_in_seq = trials[s][key_column]
        for k_offset in range(1, max_allowed_consecutive + 1):
            # s + k_offset must be < n.
            # The loop for s (up to n - (max_allowed_consecutive + 1)) ensures
            # s + max_allowed_consecutive is a valid index (i.e., < n).
            if trials[s + k_offset][key_column] != first_val_in_seq:
                is_potential_violation = False
                break
        if is_potential_violation:
            return False # Violation found involving check_idx for this key

    return True # No violation involving check_idx for this key

# --- 2. Smart Shuffling with Constraint Satisfaction ---
def shuffle_with_constraint(trials_list, max_consecutive_img, max_consecutive_type, max_shuffle_attempts=100, max_fix_passes=500):
    original_trials = trials_list.copy() 

    for shuffle_attempt in range(max_shuffle_attempts):
        current_trials = original_trials.copy()
        random.shuffle(current_trials)

        for fix_pass in range(max_fix_passes):
            violation_found_this_pass = False
            i = 0
            while i < len(current_trials): 
                
                offending_idx = -1
                violation_type_str = None # 'image' or 'type'
                current_max_consecutive_for_viol = -1 # Stores max_consecutive for the specific violation found

                # Check for IMAGE violation starting at i
                # A sequence of (max_consecutive_img + 1) identical images is a violation.
                if i <= len(current_trials) - (max_consecutive_img + 1):
                    is_viol = True
                    first_val = current_trials[i]['image_file']
                    # Check elements from i+1 to i+max_consecutive_img
                    for k in range(1, max_consecutive_img + 1): 
                        if current_trials[i+k]['image_file'] != first_val:
                            is_viol = False; break
                    if is_viol:
                        violation_type_str = 'image'
                        # The (max_consecutive_img+1)-th item is at index i + max_consecutive_img
                        offending_idx = i + max_consecutive_img 
                        current_max_consecutive_for_viol = max_consecutive_img
                
                # Check for TYPE (match/mismatch) violation starting at i,
                # only if no image violation was already flagged starting at i (image violations take precedence for fixing).
                if not violation_type_str and (i <= len(current_trials) - (max_consecutive_type + 1)):
                    is_viol = True
                    first_val = current_trials[i]['is_match']
                    for k in range(1, max_consecutive_type + 1):
                        if current_trials[i+k]['is_match'] != first_val:
                            is_viol = False; break
                    if is_viol:
                        violation_type_str = 'type'
                        offending_idx = i + max_consecutive_type
                        current_max_consecutive_for_viol = max_consecutive_type

                if violation_type_str: # A violation (either image or type) was found starting at i
                    violation_found_this_pass = True
                    found_swap = False
                    
                    # Try to find a suitable swap target for current_trials[offending_idx]
                    search_indices = list(range(offending_idx + 1, len(current_trials))) + list(range(i))
                    random.shuffle(search_indices) # Shuffle to avoid predictable swaps

                    for j in search_indices:
                        # Condition 1: Swap target (item at j) must differ on the violating property
                        # to help break the current sequence.
                        key_for_violation = 'image_file' if violation_type_str == 'image' else 'is_match'
                        val_at_offending_item = current_trials[offending_idx][key_for_violation]
                        val_at_j_item = current_trials[j][key_for_violation]
                        
                        if val_at_j_item == val_at_offending_item:
                            continue # This swap won't help break the current specific violation sequence

                        # Tentative swap
                        current_trials[offending_idx], current_trials[j] = current_trials[j], current_trials[offending_idx]

                        # Condition 2: The original violation at i (for violation_type_str) must be resolved by this swap.
                        original_violation_resolved = True
                        # Check if the sequence [i...i+current_max_consecutive_for_viol] is still all same for key_for_violation
                        is_still_viol_at_i = True # Assume it is, prove it's not
                        first_val_after_swap_at_i = current_trials[i][key_for_violation]
                        for k_check in range(1, current_max_consecutive_for_viol + 1):
                            if i + k_check >= len(current_trials) or \
                               current_trials[i+k_check][key_for_violation] != first_val_after_swap_at_i:
                                is_still_viol_at_i = False # Found a different item, so sequence is broken
                                break
                        if is_still_viol_at_i: # Sequence starting at i is still violating for this key
                            original_violation_resolved = False
                        
                        # Condition 3: The swap must NOT create new violations (of EITHER type)
                        # around the two positions involved in the swap (offending_idx and j).
                        creates_new_violation = False
                        if not original_violation_resolved:
                            pass # No need to check further if main issue not fixed
                        else:
                            # Check around the new item at offending_idx (which was current_trials[j] before swap)
                            if not check_local_validity(current_trials, offending_idx, 'image_file', max_consecutive_img): creates_new_violation = True
                            if not creates_new_violation and not check_local_validity(current_trials, offending_idx, 'is_match', max_consecutive_type): creates_new_violation = True
                            # Check around the new item at j (which was current_trials[offending_idx] before swap)
                            if not creates_new_violation and not check_local_validity(current_trials, j, 'image_file', max_consecutive_img): creates_new_violation = True
                            if not creates_new_violation and not check_local_validity(current_trials, j, 'is_match', max_consecutive_type): creates_new_violation = True
                        
                        if original_violation_resolved and not creates_new_violation:
                            found_swap = True
                            # Restart scan from an earlier position to catch cascading effects or newly formed violations.
                            i = max(0, i - current_max_consecutive_for_viol) 
                            break # Break from j loop (found a good swap)
                        else:
                            # Swap back if it didn't fix the original issue or created a new one
                            current_trials[offending_idx], current_trials[j] = current_trials[j], current_trials[offending_idx]
                    
                    if found_swap:
                        # `i` has been reset due to a successful swap.
                        # The `while` loop will continue with the new `i`.
                        # Skip the `i += 1` at the end of this `while` loop iteration.
                        continue 
                
                # If no violation was found starting at current i, OR
                # if a violation was found but no suitable swap could fix it,
                # then increment i to check the next position in the list.
                i += 1
            
            if not violation_found_this_pass:
                # print(f"  Constraints satisfied after {fix_pass + 1} fixing passes for shuffle attempt {shuffle_attempt + 1}.")
                return current_trials # Success for this shuffle attempt!

        # print(f"  Fixing failed after {max_fix_passes} passes for shuffle attempt {shuffle_attempt + 1}. Re-shuffling.")

    print(f"WARNING: Could not satisfy shuffle constraints after {max_shuffle_attempts} full shuffles with fixing. Returning last attempt (may contain violations).")
    return current_trials # Return last attempt even if not perfect

# --- 3. Main Loop to Generate Files ---
print("Generating base trials...")
base_rows = generate_base_trials(IMAGES, TRIALS_PER_IMAGE_TYPE)
print(f"Total base trials: {len(base_rows)}")
expected_matches = TRIALS_PER_IMAGE_TYPE * len(IMAGES)
expected_mismatches = TRIALS_PER_IMAGE_TYPE * len(IMAGES)
print(f"Expecting {expected_matches} match and {expected_mismatches} mismatch trials.")

generated_files_count = 0
for i_file in range(NUM_FILES_TO_GENERATE):
    file_seed = i_file + 1 
    random.seed(file_seed)
    print(f"\nGenerating file {i_file+1}/{NUM_FILES_TO_GENERATE} with seed {file_seed}...")

    trials_for_this_file = []
    for trial_template in base_rows:
        new_trial = trial_template.copy()
        new_trial['blank_duration'] = round(random.uniform(1.0, 2.0), 2)
        new_trial['iti_duration'] = round(random.uniform(1.0, 3.0), 2)
        trials_for_this_file.append(new_trial)

    shuffled_trials = shuffle_with_constraint(
        trials_for_this_file,
        MAX_CONSECUTIVE_IDENTICAL_IMAGES,
        MAX_CONSECUTIVE_MATCH_MISMATCH
    )

    if shuffled_trials:
        df = pd.DataFrame(shuffled_trials)
        
        # Verification
        violation_in_final = False
        # Check image violations
        for k_check in range(len(df) - MAX_CONSECUTIVE_IDENTICAL_IMAGES):
            is_img_viol = True
            img_f = df.iloc[k_check]['image_file']
            for V_offset in range(1, MAX_CONSECUTIVE_IDENTICAL_IMAGES + 1):
                 if df.iloc[k_check + V_offset]['image_file'] != img_f:
                    is_img_viol = False
                    break
            if is_img_viol:
                violation_in_final = True
                print(f"  WARNING: File {i_file+1} has IMAGE violation at index {k_check}:")
                print(df.iloc[k_check : k_check + MAX_CONSECUTIVE_IDENTICAL_IMAGES + 1]['image_file'])
                break 
        
        # Check type (match/mismatch) violations if no image violation found yet
        if not violation_in_final:
            for k_check in range(len(df) - MAX_CONSECUTIVE_MATCH_MISMATCH):
                is_type_viol = True
                type_f = df.iloc[k_check]['is_match']
                for V_offset in range(1, MAX_CONSECUTIVE_MATCH_MISMATCH + 1):
                     if df.iloc[k_check + V_offset]['is_match'] != type_f:
                        is_type_viol = False
                        break
                if is_type_viol:
                    violation_in_final = True
                    print(f"  WARNING: File {i_file+1} has TYPE (match/mismatch) violation at index {k_check}:")
                    print(df.iloc[k_check : k_check + MAX_CONSECUTIVE_MATCH_MISMATCH + 1]['is_match'])
                    break

        if not violation_in_final:
            print(f"  File {i_file+1} successfully generated and constraints met.")
            generated_files_count += 1
        else:
            print(f"  File {i_file+1} generated WITH VIOLATIONS after all attempts.")

        file_path = os.path.join(OUTPUT_DIR, f"localizer_conditions_{i_file+1}.csv")
        df.to_csv(file_path, index=False)
    else:
        # This case should ideally not be reached if shuffle_with_constraint always returns a list
        print(f"  Failed to generate a trial list for file {i_file+1} (shuffled_trials was empty/None).")


print(f"\nGenerated {generated_files_count} files that met constraints out of {NUM_FILES_TO_GENERATE} requested.")

# --- Final Check on one of the generated DataFrames (optional) ---
if generated_files_count > 0 and NUM_FILES_TO_GENERATE > 0:
    # Try to check the last successfully generated file, or the last requested if all failed.
    # For simplicity, just checking the file named with NUM_FILES_TO_GENERATE index.
    last_file_idx = NUM_FILES_TO_GENERATE 
    last_file_path = os.path.join(OUTPUT_DIR, f"localizer_conditions_{last_file_idx}.csv")
    try:
        last_df = pd.read_csv(last_file_path)
        match_count = last_df[last_df['is_match'] == 'match'].shape[0]
        mismatch_count = last_df[last_df['is_match'] == 'mismatch'].shape[0]
        total_trials = len(last_df)
        print(f"\n--- Example check on last generated file ({os.path.basename(last_file_path)}) ---")
        print(f"Total trials: {total_trials}")
        print(f"Match trials: {match_count} (Expected: {expected_matches})")
        print(f"Mismatch trials: {mismatch_count} (Expected: {expected_mismatches})")

        print("\nVerifying IMAGE constraint (no more than MAX_CONSECUTIVE_IDENTICAL_IMAGES in a row):")
        image_violation_found = False
        for i in range(len(last_df) - MAX_CONSECUTIVE_IDENTICAL_IMAGES):
            imgs_to_check = [last_df.iloc[i+k]['image_file'] for k in range(MAX_CONSECUTIVE_IDENTICAL_IMAGES + 1)]
            if len(set(imgs_to_check)) == 1:
                print(f"IMAGE VIOLATION FOUND at index {i}: {imgs_to_check}")
                image_violation_found = True
        if not image_violation_found:
            print("Image constraint successfully met in the example file.")
        else:
            print("IMAGE CONSTRAINT VIOLATED in the example file.")

        print("\nVerifying TYPE (match/mismatch) constraint (no more than MAX_CONSECUTIVE_MATCH_MISMATCH in a row):")
        type_violation_found = False
        for i in range(len(last_df) - MAX_CONSECUTIVE_MATCH_MISMATCH):
            types_to_check = [last_df.iloc[i+k]['is_match'] for k in range(MAX_CONSECUTIVE_MATCH_MISMATCH + 1)]
            if len(set(types_to_check)) == 1:
                print(f"TYPE VIOLATION FOUND at index {i}: {types_to_check}")
                type_violation_found = True
        if not type_violation_found:
            print("Type (match/mismatch) constraint successfully met in the example file.")
        else:
            print("TYPE (match/mismatch) CONSTRAINT VIOLATED in the example file.")

    except FileNotFoundError:
        print(f"Could not find {last_file_path} for final check.")
elif NUM_FILES_TO_GENERATE > 0:
    print(f"\nNo files were successfully generated that met constraints, so final check cannot be performed.")


