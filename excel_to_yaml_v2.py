import pandas as pd
import yaml
import sys

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'troubleshooting_logic.xlsx' 
OUTPUT_FILE = 'WM_generated_flow.yaml'

def extract_trigger(logic_string):
    """Extracts the 'Key' from logic strings like 'Key=Value'"""
    if isinstance(logic_string, str) and '=' in logic_string:
        return logic_string.split('=', 1)[0].strip()
    return None

def generate_yaml():
    try:
        # 1. Load Data
        print(f"Reading {INPUT_FILE}...")
        # Ensure we read 'Key' and 'Value' columns for Metadata correctly
        meta_df = pd.read_excel(INPUT_FILE, sheet_name='Metadata')
        if 'Key' in meta_df.columns:
            meta_df = meta_df.set_index('Key')
        
        steps_df = pd.read_excel(INPUT_FILE, sheet_name='Steps').fillna('')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 2. Build Metadata Block
    # Helper to safely get metadata values
    def get_meta(key, default=''):
        try:
            return str(meta_df.loc[key, 'Value'])
        except KeyError:
            return default

    yaml_structure = {
        'meta': {
            'id': get_meta('id', 'WM_TEST_VIBRATION_V1'),
            'title': get_meta('title', 'Diagnostic Flow'),
            'version': get_meta('version', '1'),
            'language': ['en'],
            'gating': {
                'require_all_steps': True,
                'generate_token': True,
                'token_pattern': get_meta('token_pattern', '{FAULT}-{SKU}-{RAND5}')
            }
        },
        'start': get_meta('start_step', 'step1_check_legs'),
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

        # --- SMART OPTION DETECTION ---
        # 1. Start with explicit Options column
        options_set = []
        options_raw = str(row['Options']).strip()
        if options_raw and options_raw.lower() != 'nan':
            options_set.extend([opt.strip() for opt in options_raw.split('|')])

        # 2. Scrape options from Logic Columns (Parts, Branch Logic, Default Next)
        #    If a column contains "Option=Target", "Option" must be a button.
        logic_columns = ['Branch Logic', 'Default Next', 'Parts']
        for col in logic_columns:
            if col in row:
                val = str(row[col]).strip()
                if '=' in val:
                    # Handle multiple logic entries split by |
                    sub_logics = val.split('|')
                    for logic in sub_logics:
                        trigger = extract_trigger(logic)
                        if trigger and trigger not in options_set:
                            options_set.append(trigger)
        
        # 3. Deduplicate (preserving order)
        final_options = []
        for opt in options_set:
            if opt not in final_options:
                final_options.append(opt)

        if final_options:
            step_block['ui'] = {
                'control': 'radio',
                'options': final_options
            }
        else:
            step_block['ui'] = {'control': 'none'}

        # --- Handle Branching Logic ---
        # Look for explicit branching instructions
        branches_raw = str(row.get('Branch Logic', '')).strip()
        if branches_raw and branches_raw.lower() != 'nan':
            branches_list = []
            logic_parts = branches_raw.split('|')
            for logic in logic_parts:
                if '=' in logic:
                    trigger, target = logic.split('=', 1)
                    branches_list.append({
                        'when': f"selection == '{trigger.strip()}'",
                        'next': target.strip()
                    })
            if branches_list:
                step_block['branches'] = branches_list

        # --- Handle Evidence/Photo ---
        if str(row.get('Capture', '')).lower() == 'photo':
            step_block['evidence'] = {
                'required': True,
                'capture': 'photo',
                'instructions': {'en': f"Capture photo for {step_id}"}
            }

        # --- Handle Parts ---
        # The 'Parts' column often contains the routing logic in this file format
        part_raw = str(row.get('Parts', '')).strip()
        if part_raw and part_raw.lower() != 'nan':
            step_block['recommends_part'] = part_raw
            step_block['recommends_parts'] = [part_raw]

        # --- Handle Resolution Prompt ---
        res_prompt = str(row.get('Resolve Prompt', '')).upper()
        if res_prompt == 'FALSE':
            step_block['resolution_prompt'] = False
        elif row['Type'] == 'quiz' and res_prompt != 'TRUE':
             step_block['resolution_prompt'] = False

        step_block['require_pass'] = True

        # --- Handle Next Step ---
        default_next = str(row.get('Default Next', '')).strip()
        if default_next and default_next.lower() != 'nan':
            step_block['next'] = default_next
        else:
            step_block['next'] = None

        yaml_structure['steps'].append(step_block)

    # 4. Write to File
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data): return True

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
        file.write(f"# Auto-generated from {INPUT_FILE}\n")
        yaml.dump(yaml_structure, file, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True, width=1000)

    print(f"✅ Success! YAML file created at: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_yaml()