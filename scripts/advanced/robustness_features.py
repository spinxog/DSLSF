try:
            is_planar, _ = nx.check_planarity(contact_graph)
            return is_planar
        except Exception as e:
            # For large graphs, use edge count estimation
            return contact_graph.number_of_edges() <= 3 * contact_graph.number_of_nodes() - 6