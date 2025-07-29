#Create conditions files for learning phase
#we have 3 stimuli : A, B, C
#We want to randomized which stimuli is A, B, C,
#We then want to present A-B, B-C, C-A
#For this experiment we added a second sequence 1,2,3
    
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)


import pandas as pd
import random
import os

# name the 6 stimuli


zebra = 'stimuli/zèbre.png'
face = 'stimuli/face.png'
banana = 'stimuli/banane.png'
scissor = 'stimuli/ciseau.png'
violon = 'stimuli/violon.png'
lunette = 'stimuli/lunette.png'


#Create 2 list of 3 stimuli (distinct stimuli)

stimuli = [zebra, face, banana, scissor, violon, lunette]

#We will randomly split the stimuli into 2 list and shuffle them 

for i in range(1, 100):
    # Randomly shuffle the stimuli
    random.seed(i)
    random.shuffle(stimuli)

    #split list in 2
    stimuli_1 = stimuli[:3]
    stimuli_2 = stimuli[3:]

    #Change data frame so that we have 3 rows in csv, pair_name, stim1_img and stim2_img
    df_list1 = pd.DataFrame({
        'pair_name': ['A_B', 'B_C', 'C_A'],
        'stim1_img': [stimuli_1[0], stimuli_1[1], stimuli_1[2]],
        'stim2_img': [stimuli_1[1], stimuli_1[2], stimuli_1[0]]
    })

    df_list2 = pd.DataFrame({
        'pair_name': ['1_2', '2_3', '3_1'],
        'stim1_img': [stimuli_2[0], stimuli_2[1], stimuli_2[2]],
        'stim2_img': [stimuli_2[1], stimuli_2[2], stimuli_2[0]]
    })

    #concat both data frames into a single one
    df = pd.concat([df_list1, df_list2])
    # Save the DataFrame to a CSV file
    

    
    output_dir = 'conditions_learning'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f'learning_conditions_{i}.csv'), index=False)


    