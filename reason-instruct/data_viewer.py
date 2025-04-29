#!/usr/bin/env python3
import os
import json
import gradio as gr
import pandas as pd
import glob
from datetime import datetime

# Constants
PROCESSED_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed_results")

def load_jsonl_file(file_path):
    """Load data from a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_available_files():
    """Get a list of available JSONL files in the processed results directory."""
    files = glob.glob(os.path.join(PROCESSED_RESULTS_DIR, "*.jsonl"))
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    # Return just the filenames without the full path
    return [os.path.basename(f) for f in files]

def load_dataset(filename):
    """Load a dataset by filename."""
    file_path = os.path.join(PROCESSED_RESULTS_DIR, filename)
    return load_jsonl_file(file_path)

def get_dataset_stats(data):
    """Calculate some basic statistics about the dataset."""
    total = len(data)
    
    # Count satisfied instructions
    satisfied = sum(1 for item in data if item.get("all_instructions_satisfied", False))
    
    # Count refinements
    total_refinements = sum(item.get("num_refinements", 0) for item in data)
    
    # Count by source dataset
    sources = {}
    for item in data:
        source = item.get("source_dataset", "unknown")
        sources[source] = sources.get(source, 0) + 1
    
    # Format the stats as a string
    stats = f"Total examples: {total}\n"
    stats += f"Satisfied instructions: {satisfied} ({satisfied/total:.1%})\n"
    stats += f"Total refinements: {total_refinements}\n"
    stats += f"Average refinements per example: {total_refinements/total:.2f}\n\n"
    stats += "Source datasets:\n"
    for source, count in sources.items():
        stats += f"- {source}: {count} ({count/total:.1%})\n"
    
    return stats

def display_example(data, index):
    """Format a single example for display."""
    if not data or index >= len(data):
        return "", "", "", "", "", ""
    
    example = data[index]
    
    # Basic info
    user_query = example.get("user_query", "N/A")
    final_answer = example.get("final_answer", "N/A")
    
    # Instructions
    instructions = example.get("atomic_instructions", [])
    instructions_text = "No instructions found"
    if instructions:
        instructions_text = "\n\n".join([f"Instruction {i+1}: {instr}" 
                                         for i, instr in enumerate(instructions)])
    
    # Verification results
    verification_results = example.get("verification_results", [])
    verification_text = "No verification results found"
    if verification_results:
        verification_parts = []
        for i, result in enumerate(verification_results):
            instruction = result.get("instruction", "N/A")
            satisfied = result.get("satisfied", False)
            explanation = result.get("explanation", "No explanation")
            status = "✅ SATISFIED" if satisfied else "❌ NOT SATISFIED"
            verification_parts.append(f"Instruction {i+1}: {instruction}\n{status}\nExplanation: {explanation}")
        verification_text = "\n\n".join(verification_parts)
    
    # Refinement trace
    trace = example.get("reasoning_trace", [])
    trace_text = "No reasoning trace found"
    if trace:
        trace_text = "\n\n".join([f"Step {i+1}:\n{step}" for i, step in enumerate(trace)])
    
    # Stats
    num_refinements = example.get("num_refinements", 0)
    all_satisfied = example.get("all_instructions_satisfied", False)
    source = example.get("source_dataset", "unknown")
    
    stats = f"Source: {source}\n"
    stats += f"Number of refinements: {num_refinements}\n"
    stats += f"All instructions satisfied: {'Yes ✅' if all_satisfied else 'No ❌'}\n"
    
    return user_query, instructions_text, final_answer, verification_text, trace_text, stats

def create_dataframe(data):
    """Create a pandas DataFrame from the data for the table view."""
    if not data:
        return pd.DataFrame()
    
    # Extract relevant fields for the table
    table_data = []
    for i, item in enumerate(data):
        table_data.append({
            "Index": i,
            "Source": item.get("source_dataset", "unknown"),
            "Instructions Satisfied": "Yes ✅" if item.get("all_instructions_satisfied", False) else "No ❌",
            "Refinements": item.get("num_refinements", 0),
            "Query Preview": item.get("user_query", "")[:50] + "..." if len(item.get("user_query", "")) > 50 else item.get("user_query", ""),
        })
    
    df = pd.DataFrame(table_data)
    # Convert Index to string to ensure it's preserved in the selection
    df['Index'] = df['Index'].astype(str)
    return df

def update_table(data):
    """Update the table view."""
    df = create_dataframe(data)
    if df.empty:
        return pd.DataFrame()
    return df

def update_view(data, selected_data):
    """Update the detail view based on selected table row."""
    if not data or selected_data is None or len(selected_data) == 0:
        return "", "", "", "", "", ""
    
    # Get the index from the selected data
    try:
        index = int(selected_data[0]['Index'])
        return display_example(data, index)
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error processing selection: {e}, {selected_data}")
        return "", "", "", "", "", ""

def load_and_update(filename):
    """Load a dataset and update all UI components."""
    data = load_dataset(filename)
    stats = get_dataset_stats(data)
    df = update_table(data)
    # Display the first example by default
    example_display = display_example(data, 0) if data else ("", "", "", "", "", "")
    
    return data, stats, df, *example_display

def navigate_example(data, current_idx, direction):
    """Navigate to the previous or next example."""
    if not data:
        return current_idx, "", "", "", "", "", ""
    
    # Calculate new index
    max_idx = len(data) - 1
    new_idx = current_idx + direction
    
    # Wrap around if needed
    if new_idx < 0:
        new_idx = max_idx
    elif new_idx > max_idx:
        new_idx = 0
    
    # Get example data
    example_data = display_example(data, new_idx)
    
    # Return new index and example data
    return new_idx, *example_data

def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="Reasoning Datasets Viewer") as app:
        # Store the current dataset and index in state
        current_data = gr.State([])
        current_idx = gr.State(0)
        
        gr.Markdown("# Reasoning Instructions Dataset Viewer")
        
        with gr.Row():
            with gr.Column(scale=1):
                # File selector
                file_dropdown = gr.Dropdown(
                    choices=get_available_files(),
                    label="Select Dataset File",
                    value=get_available_files()[0] if get_available_files() else None
                )
                
                # Dataset stats
                stats_text = gr.Textbox(label="Dataset Statistics", lines=10, interactive=False)
                
                # Load button
                load_btn = gr.Button("Load Dataset")
            
            with gr.Column(scale=3):
                # Table view of examples (read-only)
                examples_table = gr.DataFrame(
                    label="Examples Overview",
                    interactive=False
                )
        
        # Navigation controls
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                prev_btn = gr.Button("⬅️ Previous")
            
            with gr.Column(scale=1, min_width=100):
                example_counter = gr.Textbox(label="Current Example", value="0 / 0", interactive=False)
            
            with gr.Column(scale=1, min_width=100):
                next_btn = gr.Button("Next ➡️")
        
        # Detail view for selected example
        with gr.Tabs():
            with gr.TabItem("User Query"):
                query_text = gr.Textbox(label="User Query", lines=5, interactive=False)
            
            with gr.TabItem("Atomic Instructions"):
                instructions_text = gr.Textbox(label="Atomic Instructions", lines=10, interactive=False)
            
            with gr.TabItem("Final Answer"):
                answer_text = gr.Textbox(label="Final Answer", lines=10, interactive=False)
            
            with gr.TabItem("Verification Results"):
                verification_text = gr.Textbox(label="Verification Results", lines=15, interactive=False)
            
            with gr.TabItem("Reasoning Trace"):
                trace_text = gr.Textbox(label="Reasoning Trace", lines=20, interactive=False)
            
            with gr.TabItem("Example Stats"):
                example_stats_text = gr.Textbox(label="Example Statistics", lines=5, interactive=False)
        
        # Function to update example counter
        def update_counter(data, idx):
            if not data:
                return "0 / 0"
            return f"{idx + 1} / {len(data)}"
        
        # Load dataset and update UI when load button is clicked
        load_btn.click(
            fn=lambda filename: (*load_and_update(filename), 0),
            inputs=[file_dropdown],
            outputs=[
                current_data, stats_text, examples_table, 
                query_text, instructions_text, answer_text, 
                verification_text, trace_text, example_stats_text,
                current_idx
            ]
        ).then(
            fn=update_counter,
            inputs=[current_data, current_idx],
            outputs=[example_counter]
        )
        
        # Navigate to previous example
        prev_btn.click(
            fn=navigate_example,
            inputs=[current_data, current_idx, gr.Number(value=-1, visible=False)],
            outputs=[current_idx, query_text, instructions_text, answer_text, 
                     verification_text, trace_text, example_stats_text]
        ).then(
            fn=update_counter,
            inputs=[current_data, current_idx],
            outputs=[example_counter]
        )
        
        # Navigate to next example
        next_btn.click(
            fn=navigate_example,
            inputs=[current_data, current_idx, gr.Number(value=1, visible=False)],
            outputs=[current_idx, query_text, instructions_text, answer_text, 
                     verification_text, trace_text, example_stats_text]
        ).then(
            fn=update_counter,
            inputs=[current_data, current_idx],
            outputs=[example_counter]
        )
        
    return app

if __name__ == "__main__":
    # Create and launch the interface
    app = create_interface()
    app.launch(show_error=True)
