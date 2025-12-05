import pandas as pd
import yaml
import sys

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'troubleshooting_logic.xlsx' 
OUTPUT_FILE = 'WM_generated_flow.yaml'

def extract_target_step(logic_string):
    """
    Parses strings like 'Option=TargetStep' and returns 'TargetStep'.
    If no '=', returns the string as is.
    """
    if isinstance(logic_string, str) and '=' in logic_string:
        return logic_string.split('=', 1)[1].strip()
    return str(logic_string).strip()

def extract_trigger(logic_string):
    """Parses strings like 'Option=TargetStep' and returns 'Option'."""
    if isinstance(logic_string, str) and '=' in logic_string:
        return logic_string.split('=', 1)[0].strip()
    return None

def generate_yaml():
    try:
        print(f"Reading {INPUT_FILE}...")
        meta_df = pd.read_excel(INPUT_FILE, sheet_name='Metadata')
        if 'Key' in meta_df.columns:
            meta_df = meta_df.set_index('Key')
        
        steps_df = pd.read_excel(INPUT_FILE, sheet_name='Steps').fillna('')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Helper to get metadata
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

    for index, row in steps_df.iterrows():
        step_id = row['ID']
        if not step_id: continue 

        step_block = {
            'id': step_id,
            'type': row['Type'],
            'prompt': {'en': row['Prompt']}
        }

        # --- OPTION DETECTION ---
        options_set = []
        options_raw = str(row['Options']).strip()
        if options_raw and options_raw.lower() != 'nan':
            options_set.extend([opt.strip() for opt in options_raw.split('|')])

        # Infer options from Logic Columns
        logic_columns = ['Branch Logic', 'Default Next', 'Parts']
        for col in logic_columns:
            if col in row:
                val = str(row[col]).strip()
                if '=' in val:
                    sub_logics = val.split('|')
                    for logic in sub_logics:
                        trigger = extract_trigger(logic)
                        if trigger and trigger not in options_set:
                            options_set.append(trigger)
        
        # Deduplicate
        final_options = []
        for opt in options_set:
            if opt not in final_options:
                final_options.append(opt)

        if final_options:
            step_block['ui'] = {'control': 'radio', 'options': final_options}
        else:
            step_block['ui'] = {'control': 'none'}

        # --- Parts Logic ---
        part_raw = str(row.get('Parts', '')).strip()
        if part_raw and part_raw.lower() != 'nan':
            step_block['recommends_part'] = part_raw
            step_block['recommends_parts'] = [part_raw]

        # --- Evidence ---
        if str(row.get('Capture', '')).lower() == 'photo':
            step_block['evidence'] = {
                'required': True,
                'capture': 'photo',
                'instructions': {'en': f"Capture photo for {step_id}"}
            }

        step_block['require_pass'] = True
        
        # --- FIX: CLEAN NEXT STEP ---
        # This is where the fix happens. We use extract_target_step to remove "Yes="
        default_next_raw = str(row.get('Default Next', '')).strip()
        
        if default_next_raw and default_next_raw.lower() != 'nan':
            # If there's a pipe |, take the first one (or handle logic if you prefer)
            # For 'next', usually we just want the ID.
            if '|' in default_next_raw:
                 default_next_raw = default_next_raw.split('|')[0]
            
            clean_next_id = extract_target_step(default_next_raw)
            step_block['next'] = clean_next_id
        else:
            step_block['next'] = None

        yaml_structure['steps'].append(step_block)

    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data): return True

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
        file.write(f"# Auto-generated from {INPUT_FILE}\n")
        yaml.dump(yaml_structure, file, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True, width=1000)

    print(f"✅ Success! YAML file created at: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_yaml()