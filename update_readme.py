
import os
import re
import glob
import sys

def get_latest_battery_run(data_root):
    """
    Finds the most recent 'Standard Battery' run in data/.
    Criteria:
    - Not 'media'
    - Not starting with 'UIT_'
    - Contains 'benchmark_*.png' images directly or in a nested folder.
    """
    subdirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    valid_runs = []
    
    for d in subdirs:
        dirname = os.path.basename(d)
        if dirname.lower() == 'media': continue
        if dirname.startswith('UIT_'): continue
        
        # Check for images in d or d/dirname (double nesting)
        images = []
        content_dir = d
        
        # Check direct
        direct_images = glob.glob(os.path.join(d, 'benchmark_*.png'))
        if direct_images:
            images = direct_images
            content_dir = d
        else:
            # Check nested
            nested_path = os.path.join(d, dirname)
            if os.path.isdir(nested_path):
                nested_images = glob.glob(os.path.join(nested_path, 'benchmark_*.png'))
                if nested_images:
                    images = nested_images
                    content_dir = nested_path
                    
        if images:
            # CHECK: Ensure this is a "Battery" run by checking for SUFFICIENT benchmarks
            # If it only has mod17 (1 image), it's not the battery the user wants.
            # User Criteria: "more than 3 benchmarks"
            if len(images) > 3:
                # Found a valid BATTERY run
                mtime = os.path.getmtime(d)
                valid_runs.append({
                    'uid': dirname,
                    'root_path': d,
                    'content_path': content_dir,
                    'mtime': mtime,
                    'images': images
                })
            
    if not valid_runs:
        return None
        
    # Sort by mtime descending (newest first)
    valid_runs.sort(key=lambda x: x['mtime'], reverse=True)
    return valid_runs[0]

def get_latest_uit_run(data_root):
    """
    Finds the most recent 'UIT Battery' run in data/.
    Criteria:
    - Starts with 'UIT_'
    - Contains 'plots' folder with 'scientific_duel_story_*.png'
    """
    subdirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    valid_runs = []
    
    for d in subdirs:
        dirname = os.path.basename(d)
        if not dirname.startswith('UIT_'): continue
        
        # Check for plots/scientific_duel_story_*.png
        plot_dir = os.path.join(d, 'plots')
        if not os.path.exists(plot_dir): continue
        
        images = glob.glob(os.path.join(plot_dir, 'scientific_duel_story_*.png'))
        if images:
             mtime = os.path.getmtime(d)
             # Extract raw UID from folder name (UIT_xxxx) -> xxxx
             uid = dirname.replace("UIT_", "")
             valid_runs.append({
                'uid': uid,
                'root_path': d,
                'plot_dir': plot_dir,
                'log_dir': os.path.join(d, 'logs'),
                'mtime': mtime,
                'images': images
             })
             
    if not valid_runs:
        return None
        
    valid_runs.sort(key=lambda x: x['mtime'], reverse=True)
    return valid_runs[0]

def update_readme(readme_path, run_info):
    uid = run_info['uid']
    content_dir = run_info['content_path']
    print(f"Latest Battery Run Found: {uid}")
    
    project_root = os.path.dirname(os.path.abspath(readme_path))
    rel_path = os.path.relpath(content_dir, project_root).replace('\\', '/')
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Update Battery UID text
    uid_pattern = r'(Results are from the `)([a-f0-9]+)(` regression battery)'
    match = re.search(uid_pattern, content)
    if match:
        old_uid = match.group(2)
        if old_uid != uid:
            print(f"Updating Battery UID: {old_uid} -> {uid}")
            content = re.sub(uid_pattern, f'\\g<1>{uid}\\g<3>', content)
            
    # 2. Update Battery Images
    bench_map = {
        'benchmark_phase_lock': r'benchmark_phase_lock_.*\.png',
        'benchmark_parity': r'benchmark_parity_.*\.png',
        'benchmark_topology': r'benchmark_topology_.*\.png',
        'benchmark_brackets_masked': r'benchmark_brackets_masked_.*\.png',
        'benchmark_oracle': r'benchmark_oracle_.*\.png',
        'benchmark_ascii': r'benchmark_ascii_.*\.png',
        'benchmark_perms_vs_gru': r'benchmark_perms_vs_gru_.*\.png',
        'benchmark_colors': r'benchmark_colors_.*\.png',
    }
    
    files_in_run = os.listdir(content_dir)
    for key, file_pattern in bench_map.items():
        candidates = [f for f in files_in_run if re.match(file_pattern, f)]
        if not candidates: continue
        new_filename = candidates[0]
        new_image_path = f"{rel_path}/{new_filename}"
        link_pattern = r'\(data/[^)]*?/' + key + r'_[^)]*?\.png\)'
        
        if re.search(link_pattern, content):
            content = re.sub(link_pattern, f'({new_image_path})', content)

    if content != original_content:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("README.md updated for Battery.")
    else:
        print("README.md Battery section is up to date.")

def update_readme_uit(readme_path, run_info):
    uid = run_info['uid']
    plot_dir = run_info['plot_dir']
    log_dir = run_info['log_dir']
    root_path = run_info['root_path']
    print(f"Latest UIT Run Found: {uid}")
    
    project_root = os.path.dirname(os.path.abspath(readme_path))
    rel_plot = os.path.relpath(plot_dir, project_root).replace('\\', '/')
    rel_log = os.path.relpath(log_dir, project_root).replace('\\', '/')
    rel_root = os.path.relpath(root_path, project_root).replace('\\', '/')
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Update UIT Header UID
    # Pattern: `## 8. Live Crystalline Verification (UID: efe843c7)`
    header_pattern = r'(Live Crystalline Verification \(UID: )([a-f0-9]+)(\))'
    if re.search(header_pattern, content):
        content = re.sub(header_pattern, f'\\g<1>{uid}\\g<3>', content)
        
    # 2. Update Scientific Story Image
    # Pattern: `(data/UIT_[^)]*?/plots/scientific_duel_story_[^)]*?\.png)`
    story_pattern = r'\(data/UIT_[^)]*?/plots/scientific_duel_story_[^)]*?\.png\)'
    new_story_path = f"{rel_plot}/scientific_duel_story_{uid}.png"
    if re.search(story_pattern, content):
        content = re.sub(story_pattern, f'({new_story_path})', content)
        
    # 3. Update Log Link
    # Pattern: `(data/UIT_[^)]*?/logs/scientific_duel_[^)]*?\.txt)`
    log_pattern = r'\(data/UIT_[^)]*?/logs/scientific_duel_[^)]*?\.txt\)'
    new_log_path = f"{rel_log}/scientific_duel_{uid}.txt"
    if re.search(log_pattern, content):
        content = re.sub(log_pattern, f'({new_log_path})', content)
        
    # 4. Update Full Data Directory Link
    # Pattern: `**Full Data Directory**: [data/UIT_efe843c7](data/UIT_efe843c7)`
    # The link part: `](data/UIT_[^)]*)`
    dir_pattern = r'(\*\*Full Data Directory\*\*: \[data/UIT_)([a-f0-9]+)(\]\(data/UIT_)([a-f0-9]+)(\))'
    # Groups: 1=prefix, 2=old_uid_text, 3=mid, 4=old_uid_link, 5=suffix
    if re.search(dir_pattern, content):
        content = re.sub(dir_pattern, f'\\g<1>{uid}\\g<3>{uid}\\g<5>', content)

    if content != original_content:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("README.md updated for UIT.")
    else:
        print("README.md UIT section is up to date.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    readme_file = os.path.join(base_dir, 'README.md')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Update both
    latest_battery = get_latest_battery_run(data_dir)
    if latest_battery:
        update_readme(readme_file, latest_battery)
    
    latest_uit = get_latest_uit_run(data_dir)
    if latest_uit:
        update_readme_uit(readme_file, latest_uit)

