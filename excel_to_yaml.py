import pandas as pd
import yaml
import sys

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'troubleshooting_logic.xlsx'
OUTPUT_FILE = 'generated_flow.yaml'

def generate_yaml():
    try:
        # 1. Load Data
        print(f"Reading {INPUT_FILE}...")
        meta_df = pd.read_excel(INPUT_FILE, sheet_name='Metadata').set_index('Key')
        steps_df = pd.read_excel(INPUT_FILE, sheet_name='Steps').fillna('')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 2. Build Metadata Block
    yaml_structure = {
        'meta': {
            'id': str(meta_df.loc['id', 'Value']),
            'title': str(meta_df.loc['title', 'Value']),
            'version': str(meta_df.loc['version', 'Value']),
            'language': ['en'],
            'gating': {
                'require_all_steps': True,
                'generate_token': True,
                'token_pattern': "{FAULT}-{SKU}-{RAND5}"
            }
        },
        'start': str(meta_df.loc['start_step', 'Value']),
        'steps': []
    }

    # 3. Iterate Through Excel Rows to Build Steps
    for index, row in steps_df.iterrows():
        step_id = row['ID']
        if not step_id: continue # Skip empty rows

        step_block = {
            'id': step_id,
            'type': row['Type'],
            'prompt': {'en': row['Prompt']}
        }

        # --- Handle UI/Options ---
        options_raw = str(row['Options']).strip()
        if options_raw:
            options_list = [opt.strip() for opt in options_raw.split('|')]
            # Determine control type based on option count or step type
            control_type = 'radio'
            if row['Type'] == 'action': 
                # Actions often imply a checkbox confirmation or radio result
                control_type = 'radio' 
            
            step_block['ui'] = {
                'control': control_type,
                'options': options_list
            }
        else:
            step_block['ui'] = {'control': 'none'}

        # --- Handle Branching Logic ---
        # Syntax in Excel: "Option Text=step_id | Other Option=other_id"
        branches_raw = str(row['Branch Logic']).strip()
        if branches_raw:
            branches_list = []
            logic_parts = branches_raw.split('|')
            for logic in logic_parts:
                if '=' in logic:
                    trigger, target = logic.split('=', 1)
                    branches_list.append({
                        'when': f"selection == '{trigger.strip()}'",
                        'next': target.strip()
                    })
                elif ' and ' in logic: # Handle complex checkbox logic if needed
                     branches_list.append({
                        'when': logic.strip(), # direct pass-through
                        'next': row['Default Next']
                    })
            
            if branches_list:
                step_block['branches'] = branches_list

        # --- Handle Evidence/Photo ---
        if str(row['Capture']).lower() == 'photo':
            step_block['evidence'] = {
                'required': True,
                'capture': 'photo',
                'instructions': {'en': f"Capture photo for {step_id}"}
            }

        # --- Handle Parts ---
        part_name = str(row['Parts']).strip()
        if part_name:
            step_block['recommends_part'] = part_name
            step_block['recommends_parts'] = [part_name]

        # --- Handle Resolution Prompt (The Annoying Popup) ---
        # Default to FALSE if not specified for quizzes
        res_prompt = str(row['Resolve Prompt']).upper()
        if res_prompt == 'FALSE':
            step_block['resolution_prompt'] = False
        elif res_prompt == 'TRUE':
            # Default behavior is usually true, but we can force it
            pass 
        elif row['Type'] == 'quiz':
            # Auto-disable for quizzes if left blank
            step_block['resolution_prompt'] = False

        # --- Handle Validation ---
        step_block['require_pass'] = True

        # --- Handle Next Step (Daisy Chain Link) ---
        default_next = str(row['Default Next']).strip()
        if default_next:
            step_block['next'] = default_next
        else:
            step_block['next'] = None

        yaml_structure['steps'].append(step_block)

    # 4. Write to File
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
        # Custom header to ensure comments or formatting if needed
        file.write(f"# Auto-generated from {INPUT_FILE}\n")
        yaml.dump(yaml_structure, file, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True, width=1000)

    print(f"✅ Success! YAML file created at: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_yaml()