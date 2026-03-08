# /// script
# dependencies = [
#   "python-pptx",
#   "matplotlib",
# ]
# ///

import os
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

# Get the directory containing the visualizations (relative to script location)
script_dir = Path(__file__).parent
# Go up to project root (from scripts/analysis/ to project root)
project_root = script_dir.parent.parent
viz_dir = project_root / "visualization" / "viz_exp4_final"
data_dir = project_root / "extracted_ascii_maps" / "paths_exp4"
levels_dir = project_root / "src" / "data" / "levels" / "exp4"
icons_dir = project_root / "public" / "icons"

# Verify the directories exist
if not viz_dir.exists():
    print(f"Error: Visualization directory not found at {viz_dir}")
    exit(1)

if not data_dir.exists():
    print(f"Error: Data directory not found at {data_dir}")
    exit(1)

# Load the agent count data
data_file = data_dir / "steps_dict_exp4_padded_new_maps.json"
if not data_file.exists():
    print(f"Error: Data file not found at {data_file}")
    exit(1)

with open(data_file, 'r') as f:
    agent_data = json.load(f)

# Function to extract agent types from level files
def extract_agent_types(level_name):
    """Extract agent types (Novice/Expert) from level TypeScript files"""
    level_file = levels_dir / f"{level_name}.ts"
    if not level_file.exists():
        print(f"Warning: Level file not found: {level_file}")
        return None

    with open(level_file, 'r') as f:
        content = f.read()

    # Extract types for each agent and experience level
    # Agent 1 in the file corresponds to agent2 in presentation
    # Agent 2 in the file corresponds to agent3 in presentation
    types = {
        'agent2': {},  # Agent 1 in file
        'agent3': {}   # Agent 2 in file
    }

    # Pattern to match experienced1/2/3 sections and their type
    for exp_num in [1, 2, 3]:
        exp_key = f'experienced{exp_num}'

        # Find agent 1 (agent2 in presentation) type
        # Match pattern: "1: { ... experienced1: { ... type: 'Novice/Expert'"
        agent1_pattern = rf"1:\s*\{{.*?{exp_key}:\s*\{{.*?type:\s*['\"](\w+)['\"]"
        agent1_match = re.search(agent1_pattern, content, re.DOTALL)
        if agent1_match:
            types['agent2'][exp_key] = agent1_match.group(1)

        # Find agent 2 (agent3 in presentation) type
        agent2_pattern = rf"2:\s*\{{.*?{exp_key}:\s*\{{.*?type:\s*['\"](\w+)['\"]"
        agent2_match = re.search(agent2_pattern, content, re.DOTALL)
        if agent2_match:
            types['agent3'][exp_key] = agent2_match.group(1)

    return types

# Get all unique level names (smXXX format, removing _exp4)
level_files = sorted(
    viz_dir.glob("stimuli_sm*_exp4_agent1_experienced1_and_agent2_experienced1.png")
)
levels = [
    f.stem.replace("stimuli_", "").replace("_exp4", "").replace(
        "_agent1_experienced1_and_agent2_experienced1", ""
    )
    for f in level_files
]

print(f"Found {len(levels)} levels to process")

def create_bar_chart(agent2_count, agent3_count, level_name):
    """Create a bar chart and save it temporarily"""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    agents = ['Agent 2', 'Agent 3']
    counts = [agent2_count, agent3_count]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax.bar(agents, counts, color=colors)
    ax.set_ylabel('Count')
    ax.set_title(f'Agent Performance - {level_name.upper()}')
    
    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom')
    
    # Remove y-axis spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save temporarily
    temp_chart_path = script_dir / f"temp_chart_{level_name}.png"
    plt.savefig(temp_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return temp_chart_path

for level in levels:
    # Extract agent types for this level
    agent_types = extract_agent_types(level)
    if agent_types:
        print(f"Level {level} agent types: {agent_types}")
    else:
        print(f"Level {level}: Could not extract agent types")

    # Add a blank slide
    blank_slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(blank_slide_layout)

    # Add title at the top (moved higher to avoid overlap with agent icons)
    title_box = slide.shapes.add_textbox(
        Inches(0.5), Inches(0.1), Inches(9), Inches(0.6)
    )
    title_frame = title_box.text_frame
    title_frame.text = level.upper()

    # Format title
    title_paragraph = title_frame.paragraphs[0]
    title_paragraph.alignment = PP_ALIGN.CENTER
    title_paragraph.font.size = Pt(32)
    title_paragraph.font.bold = True

    # Calculate dimensions for 3 images in a row
    image_width = Inches(3)
    image_height = Inches(3.2)  # Increased height to prevent vertical squishing
    spacing = Inches(0.25)
    total_width = (image_width * 3) + (spacing * 2)
    start_left = (Inches(10) - total_width) / 2
    top = Inches(1.0)  # Adjusted top position

    # Experience levels and scenario labels
    experience_levels = ["experienced1", "experienced2", "experienced3"]
    scenario_labels = ["Scenario 1", "Scenario 2", "Scenario 3"]

    # Collect agent counts for bar chart (use first scenario data as example)
    agent2_counts = []
    agent3_counts = []
    
    for i, exp_level in enumerate(experience_levels):
        # Add agent type info with icons above image
        if agent_types:
            exp_key = f'experienced{i+1}'
            agent2_type = agent_types['agent2'].get(exp_key, '')
            agent3_type = agent_types['agent3'].get(exp_key, '')

            if agent2_type and agent3_type:
                # Calculate positions for side-by-side layout
                column_left = start_left + (image_width + spacing) * i
                icon_size = Inches(0.25)  # Small icon size
                label_height = Inches(0.3)
                y_position = top - Inches(0.4)

                # Agent 2 (blue icon) on the left
                agent2_icon_path = icons_dir / "al.png"
                if agent2_icon_path.exists():
                    slide.shapes.add_picture(
                        str(agent2_icon_path),
                        column_left + Inches(0.1),
                        y_position,
                        width=icon_size,
                        height=icon_size
                    )

                # Agent 2 type label only (no "Agent2:" prefix)
                agent2_label_box = slide.shapes.add_textbox(
                    column_left + Inches(0.4),
                    y_position,
                    Inches(1.2),
                    label_height
                )
                agent2_label_frame = agent2_label_box.text_frame
                agent2_label_frame.text = f"{agent2_type}"
                agent2_label_frame.word_wrap = False
                agent2_para = agent2_label_frame.paragraphs[0]
                agent2_para.font.size = Pt(10)
                agent2_para.font.bold = True
                agent2_para.font.color.rgb = RGBColor(0, 100, 200)  # Blue to match agent2 icon

                # Agent 3 (green icon) on the right
                agent3_icon_path = icons_dir / "green_a.png"
                if agent3_icon_path.exists():
                    slide.shapes.add_picture(
                        str(agent3_icon_path),
                        column_left + Inches(1.65),
                        y_position,
                        width=icon_size,
                        height=icon_size
                    )

                # Agent 3 type label only (no "Agent3:" prefix)
                agent3_label_box = slide.shapes.add_textbox(
                    column_left + Inches(1.95),
                    y_position,
                    Inches(1.2),
                    label_height
                )
                agent3_label_frame = agent3_label_box.text_frame
                agent3_label_frame.text = f"{agent3_type}"
                agent3_label_frame.word_wrap = False
                agent3_para = agent3_label_frame.paragraphs[0]
                agent3_para.font.size = Pt(10)
                agent3_para.font.bold = True
                agent3_para.font.color.rgb = RGBColor(0, 150, 0)  # Green to match agent3 icon

        # Add image
        img_path = viz_dir / f"stimuli_{level}_exp4_agent1_{exp_level}_and_agent2_{exp_level}.png"
        if img_path.exists():
            left = start_left + (image_width + spacing) * i
            slide.shapes.add_picture(str(img_path), left, top, width=image_width, height=image_height)
        else:
            print(f"  Warning: Image not found: {img_path}")

        # Add scenario text below image
        scenario_box = slide.shapes.add_textbox(
            start_left + (image_width + spacing) * i,
            top + image_height + Inches(0.1),
            image_width,
            Inches(0.3)
        )
        scenario_frame = scenario_box.text_frame
        scenario_frame.text = scenario_labels[i]

        # Format scenario text
        scenario_paragraph = scenario_frame.paragraphs[0]
        scenario_paragraph.alignment = PP_ALIGN.CENTER
        scenario_paragraph.font.size = Pt(14)
        scenario_paragraph.font.color.rgb = RGBColor(89, 89, 89)
        
        # Get agent data for this scenario (if available)
        scenario_key = f"{level.lower()}_scenario{i+1}"
        if scenario_key in agent_data:
            agent2_counts.append(agent_data[scenario_key]["agent2_count"])
            agent3_counts.append(agent_data[scenario_key]["agent3_count"])
        else:
            agent2_counts.append(0)
            agent3_counts.append(0)
            print(f"  Warning: No data found for {scenario_key}")

    # Create and add bar chart below images using matplotlib for better styling
    chart_top = top + image_height + Inches(0.4)  # Reduced spacing
    chart_width = Inches(8)
    chart_height = Inches(2.0)  # Slightly smaller to fit everything
    
    # Always create the chart, even for zero values
    # This ensures cases like sm371 (all zeros) are displayed
    if True:
        # Create matplotlib chart with clean styling
        plt.style.use('default')  # Use default style to avoid shadows
        fig, ax = plt.subplots(figsize=(10, 2.5), dpi=120)  # Larger figure for bigger text

        # Styling constants from reference
        colors = ["#1f77b4", "#90ee90"]  # Blue and Light Green
        bar_width = 0.35
        alpha = 0.8

        # Prepare data
        scenarios = ['Scenario 1', 'Scenario 2', 'Scenario 3']
        x_pos = np.arange(len(scenarios))

        # Calculate max value before creating bars
        max_val = max(max(agent2_counts), max(agent3_counts))
        if max_val == 0:
            max_val = 1  # Minimum for zero-value cases to ensure visibility

        # Use simple scenario labels (agent types are shown above images now)
        scenario_labels = scenarios

        # Create bars with simple agent labels
        bars1 = ax.bar(x_pos - bar_width/2, agent2_counts, bar_width,
                      color=colors[0], alpha=alpha, label='Agent 2', linewidth=0)
        bars2 = ax.bar(x_pos + bar_width/2, agent3_counts, bar_width,
                      color=colors[1], alpha=alpha, label='Agent 3', linewidth=0)
        
        # Add value labels on top of bars with even larger text (including zeros)
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                # Position text slightly above the bar, even for zero-height bars
                text_y = max(height, 0.05 * max_val)  # Minimum height for zero bars
                ax.text(bar.get_x() + bar.get_width()/2., text_y,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=16)
        
        # Styling with larger text
        # No x-label (removed "Scenarios" title)
        ax.set_ylabel('Agent Count', fontsize=16)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenario_labels, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        # Remove top and right spines (no shadow effect)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        
        # Set y-axis limit with some padding
        ax.set_ylim(0, max_val * 1.2)
        
        # Add only horizontal grid lines (no vertical lines)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        
        # Add legend below the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
                 ncol=2, frameon=False, fontsize=14)
        
        # Adjust layout to make room for legend and icon key below
        plt.tight_layout(pad=0.5, rect=(0, 0.05, 1, 1))  # Leave space at bottom for legend and icon key
        
        # Save chart as temporary image
        temp_chart_path = script_dir / f"temp_chart_{level}.png"
        plt.savefig(temp_chart_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.2)
        plt.close()
        
        # Add the chart image to slide
        if temp_chart_path.exists():
            chart_left = (Inches(10) - chart_width) / 2  # Center the chart
            slide.shapes.add_picture(str(temp_chart_path), chart_left, chart_top, 
                                   width=chart_width, height=chart_height)
            temp_chart_path.unlink()  # Clean up temp file
            
            # Add icon key below the matplotlib legend to clarify icon mappings
            # Moved even further up and more to the left
            icon_key_top = chart_top + chart_height + Inches(0.05)  # Moved even further up
            icon_key_width = Inches(4)
            icon_key_left = Inches(0.2)  # Moved further to the left
            icon_size = Inches(0.2)
            
            # Icon key title
            icon_key_title = slide.shapes.add_textbox(
                icon_key_left,
                icon_key_top,
                icon_key_width,
                Inches(0.25)
            )
            icon_key_title_frame = icon_key_title.text_frame
            icon_key_title_frame.text = "Icon Key:"
            icon_key_title_para = icon_key_title_frame.paragraphs[0]
            icon_key_title_para.font.size = Pt(10)
            icon_key_title_para.font.bold = True
            icon_key_title_para.font.color.rgb = RGBColor(100, 100, 100)
            
            # Blue icon = Agent 2
            agent2_icon_key_top = icon_key_top + Inches(0.2)
            agent2_icon_path = icons_dir / "al.png"
            if agent2_icon_path.exists():
                slide.shapes.add_picture(
                    str(agent2_icon_path),
                    icon_key_left,
                    agent2_icon_key_top,
                    width=icon_size,
                    height=icon_size
                )
                
                agent2_key_text = slide.shapes.add_textbox(
                    icon_key_left + icon_size + Inches(0.05),
                    agent2_icon_key_top,
                    Inches(2.5),
                    Inches(0.2)
                )
                agent2_key_frame = agent2_key_text.text_frame
                agent2_key_frame.text = "= Agent 2 (Blue bars)"
                agent2_key_para = agent2_key_frame.paragraphs[0]
                agent2_key_para.font.size = Pt(9)
                agent2_key_para.font.color.rgb = RGBColor(100, 100, 100)
            
            # Green icon = Agent 3
            agent3_icon_key_top = agent2_icon_key_top + Inches(0.35)  # Added more vspace between agents
            agent3_icon_path = icons_dir / "green_a.png"
            if agent3_icon_path.exists():
                slide.shapes.add_picture(
                    str(agent3_icon_path),
                    icon_key_left,
                    agent3_icon_key_top,
                    width=icon_size,
                    height=icon_size
                )
                
                agent3_key_text = slide.shapes.add_textbox(
                    icon_key_left + icon_size + Inches(0.05),
                    agent3_icon_key_top,
                    Inches(2.5),
                    Inches(0.2)
                )
                agent3_key_frame = agent3_key_text.text_frame
                agent3_key_frame.text = "= Agent 3 (Green bars)"
                agent3_key_para = agent3_key_frame.paragraphs[0]
                agent3_key_para.font.size = Pt(9)
                agent3_key_para.font.color.rgb = RGBColor(100, 100, 100)

    print(f"Added slide for {level} (Agent2: {agent2_counts}, Agent3: {agent3_counts})")

# Save the presentation (relative to script location)
output_path = script_dir / "path_comparison_presentation.pptx"
prs.save(str(output_path))
print(f"\nPresentation saved to: {output_path}")
print(f"Total slides: {len(levels)}")

# Clean up temporary files
for level in levels:
    temp_chart = script_dir / f"temp_chart_{level}.png"
    if temp_chart.exists():
        temp_chart.unlink()