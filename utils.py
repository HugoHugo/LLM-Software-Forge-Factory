def save_agent_png_graph(agent, logging, software_forge_base_path):
    """ Saves a mermaid graph for a given angent run """
    # Save graph visualization
    try:
        png_bytes = agent.get_graph().draw_mermaid_png()
        with open(software_forge_base_path + "/graph.png", "wb") as f:
            f.write(png_bytes)
        logging.info("Graph visualization saved to graph.png")
    except Exception as e:
        logging.error(f"Failed to save graph visualization: {e}")