from graphviz import Digraph

def create_colorful_architecture():
    dot = Digraph(comment='DAgger Architecture', format='png')
    
    # Global Graph Settings for a cleaner look
    dot.attr(rankdir='TB')  # Top to Bottom
    dot.attr(splines='ortho') # Orthogonal edges
    dot.attr(nodesep='0.5')
    dot.attr(ranksep='0.5')
    
    # Node Defaults
    dot.attr('node', shape='record', style='filled', fontname='Helvetica', fontsize='12', penwidth='0')
    dot.attr('edge', color='#555555', arrowsize='0.8', penwidth='1.2')

    # --- 1. Input Layer (Blue) ---
    with dot.subgraph(name='cluster_input') as c:
        c.attr(style='invis')
        c.node('input', '{Input Layer|{64, 64, 2}|Stacked Frames}', fillcolor='#E3F2FD', fontcolor='#0D47A1')

    # --- 2. Convolutional Layers (Orange) ---
    # Using specific shapes from your screenshot
    with dot.subgraph(name='cluster_conv') as c:
        c.attr(style='invis')
        c.node('conv1', '{Conv2D_1|32 Filters|8x8, Stride 4}', fillcolor='#FFF3E0', fontcolor='#E65100')
        c.node('conv2', '{Conv2D_2|64 Filters|4x4, Stride 2}', fillcolor='#FFF3E0', fontcolor='#E65100')
        c.node('conv3', '{Conv2D_3|64 Filters|3x3, Stride 1}', fillcolor='#FFF3E0', fontcolor='#E65100')

    # --- 3. Processing Layers (Green) ---
    with dot.subgraph(name='cluster_proc') as c:
        c.attr(style='invis')
        c.node('flat', '{Flatten|1024 units}', fillcolor='#E8F5E9', fontcolor='#1B5E20')
        c.node('dense_shared', '{Dense (Shared)|512 units|ReLU}', fillcolor='#C8E6C9', fontcolor='#1B5E20')

    # --- 4. Action Heads (Purple/Pink) ---
    with dot.subgraph(name='cluster_heads') as c:
        c.attr(rank='same') # Force these to be on the same level
        c.node('head_steer', '{Steer Head|1 unit|Tanh}', fillcolor='#F3E5F5', fontcolor='#4A148C')
        c.node('head_gas', '{Gas Head|1 unit|Sigmoid}', fillcolor='#FCE4EC', fontcolor='#880E4F')
        c.node('head_brake', '{Brake Head|1 unit|Sigmoid}', fillcolor='#E0F7FA', fontcolor='#006064')

    # --- 5. Output (Grey) ---
    dot.node('concat', '{Concatenate|Output: [Steer, Gas, Brake]}', fillcolor='#F5F5F5', fontcolor='#212121', shape='note')

    # --- Connections ---
    dot.edge('input', 'conv1')
    dot.edge('conv1', 'conv2')
    dot.edge('conv2', 'conv3')
    dot.edge('conv3', 'flat')
    dot.edge('flat', 'dense_shared')
    
    # Split to heads
    dot.edge('dense_shared', 'head_steer')
    dot.edge('dense_shared', 'head_gas')
    dot.edge('dense_shared', 'head_brake')
    
    # Recombine to output
    dot.edge('head_steer', 'concat')
    dot.edge('head_gas', 'concat')
    dot.edge('head_brake', 'concat')

    # Render
    output_path = dot.render('dagger_architecture', view=False)
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    try:
        create_colorful_architecture()
    except Exception as e:
        print("Error: Graphviz executable not found. Please install via: `sudo apt-get install graphviz` or `brew install graphviz`")
        print(f"Details: {e}")
