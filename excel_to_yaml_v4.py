import pandas as pd
import yaml
import sys
import os

# ---------------------------------------------------------
# Custom YAML Configuration
# ---------------------------------------------------------
class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

def str_presenter(dumper, data):
    """
    Configures YAML to use '>' (block style) for multiline strings 
    to keep your long prompts readable.
    """
    if len(data.splitlines()) > 1 or '\n' in data:  
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='>')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def clean_nan(value):
    """Returns None if value is NaN or empty string."""
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()

def process_excel(file_path, output_path):
    print(f"Reading {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        # Load sheets
        df_meta = pd.read_excel(file_path, sheet_name='Metadata')
        df_steps = pd.read_excel(file_path, sheet_name='Steps')
        
        # CRITICAL: Forward fill Step IDs to handle blank cells in the ID column
        # (This allows you to visually group rows in Excel if you want, 
        # though explicit repetition is safer)
        df_steps['Step ID'] = df_steps['Step ID'].ffill()
        
    except Exception as e:
        print(f"Error reading Excel sheets: {e}")
        return

    # -----------------------------------------------------
    # 1. Process Metadata
    # -----------------------------------------------------
    meta_row = df_meta.iloc[0]
    gating_config = {
        'require_all_steps': bool(meta_row.get('gating_require_all_steps', True)),
        'generate_token': bool(meta_row.get('gating_generate_token', True)),
        'token_pattern': meta_row.get('gating_token_pattern', "{FAULT}-{SKU}-{RAND5}")
    }
    
    yaml_structure = {
        'meta': {
            'id': meta_row['id'],
            'title': meta_row['title'],
            'version': str(meta_row['version']),
            'language': [x.strip() for x in str(meta_row['language']).split(',')],
            'gating': gating_config
        },
        'start': meta_row['start_step_id'],
        'steps': []
    }

    # -----------------------------------------------------
    # 2. Process Steps (Grouping by Step ID)
    # -----------------------------------------------------
    # We group the dataframe by 'Step ID' to process all options for a step at once.
    grouped = df_steps.groupby('Step ID', sort=False)

    for step_id, group_df in grouped:
        # Get static data from the first row of the group
        row0 = group_df.iloc[0]
        
        step = {
            'id': step_id,
            'type': row0['Type'],
            'prompt': {'en': row0['Prompt (EN)']}
        }

        # --- UI & Options (Iterate through all rows in this group) ---
        ui_control = clean_nan(row0.get('UI Control'))
        
        if ui_control and ui_control != 'none':
            step['ui'] = {'control': ui_control}
            
            options_list = []
            branches = []
            
            for _, row in group_df.iterrows():
                opt_text = clean_nan(row.get('UI Option'))
                target = clean_nan(row.get('Target Step'))
                
                if opt_text:
                    options_list.append(opt_text)
                    
                    # If a Target Step is defined for this option, create a branch
                    if target:
                        next_node = None if target.lower() == 'null' else target
                        branches.append({
                            'when': f"selection == '{opt_text}'",
                            'next': next_node
                        })

            if options_list:
                step['ui']['options'] = options_list
            
            if branches:
                step['require_pass'] = True
                step['resolution_prompt'] = False # Defaulting to false for cleaner UI
                step['branches'] = branches

        else:
            step['ui'] = {'control': 'none'}

        # --- Evidence Block ---
        if row0.get('Evidence Required') is True:
            evidence = {'required': True, 'capture': 'photo'}
            instr = clean_nan(row0.get('Evidence Instruction'))
            if instr: 
                evidence['instructions'] = {'en': instr}
            step['evidence'] = evidence

        # --- Parts Recommendation ---
        parts_str = clean_nan(row0.get('Recommends Parts'))
        if parts_str:
            parts_list = [p.strip() for p in parts_str.split(',')]
            # If specifically one part, you can use recommends_part (legacy) or just list
            if len(parts_list) == 1:
                step['recommends_part'] = parts_list[0]
            step['recommends_parts'] = parts_list

        # --- Default Next (The Daisy Chain Enforcer) ---
        default_next = clean_nan(row0.get('Default Next'))
        if default_next:
            step['next'] = None if default_next.lower() == 'null' else default_next

        yaml_structure['steps'].append(step)

    # -----------------------------------------------------
    # Write Output
    # -----------------------------------------------------
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_structure, f, Dumper=IndentDumper, sort_keys=False, allow_unicode=True)
    
    print(f"Success! YAML generated at: {output_path}")

if __name__ == "__main__":
    process_excel('troubleshooting_logic.xlsx', 'p2o_Wash_Cycle_Not_Working_Issue_in_WM.yaml')