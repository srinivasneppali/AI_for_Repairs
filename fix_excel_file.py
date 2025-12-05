import pandas as pd

# 1. Define the corrections for the Options column
corrections = {
    'step1_check_legs': "Yes, stable|No, wobbling",
    'step1_adjust_legs': "Fixed, no vibration|Issue persists (Leg Broken)",
    'step2_check_shocks': "No, looks good|Yes, leaking/broken",
    'step3_check_springs': "No, springs are good|Yes, stretched/broken"
}

try:
    # 2. Read the existing file
    input_file = 'troubleshooting_logic.xlsx'
    print(f"Reading {input_file}...")
    steps_df = pd.read_excel(input_file, sheet_name='Steps')
    meta_df = pd.read_excel(input_file, sheet_name='Metadata')

    # 3. Apply corrections
    print("Applying fixes to Options column...")
    for idx, row in steps_df.iterrows():
        if row['ID'] in corrections:
            steps_df.at[idx, 'Options'] = corrections[row['ID']]

    # 4. Save to a new file
    output_file = 'troubleshooting_logic_corrected.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        meta_df.to_excel(writer, sheet_name='Metadata', index=False)
        steps_df.to_excel(writer, sheet_name='Steps', index=False)
    
    print(f"✅ Created clean file: {output_file}")
    print("You can now rename this to 'troubleshooting_logic.xlsx' if you wish.")

except Exception as e:
    print(f"Error: {e}")