pub mod wavegraph;
#[cfg(test)]
mod test {
    use crate::wavegraph::GraphModel;
    use crate::wavegraph::WaveModel;
    use petgraph::graph::DefaultIx;
    use std::collections::HashMap;
    use sucds::bit_vectors::BitVector;

    fn create_directed_graph() -> GraphModel<String, f32, f32, petgraph::Directed, DefaultIx> {
        #![allow(unused_variables)]
        // Specific type annotation is needed here for some reason
        let mut graph =
            GraphModel::<String, f32, f32, petgraph::Directed, DefaultIx>::new_directed();
        let v1 = graph.add_node("v1".to_string(), 1.0);
        let v2 = graph.add_node("v2".to_string(), 1.5);
        let v3 = graph.add_node("v3".to_string(), 1.0);
        let v4 = graph.add_node("v4".to_string(), 2.0);

        // Edges:
        // v1 -> v2
        let e1 = graph.add_edge(v1, v2, "e1".to_string(), 1.0);
        // v1 -> v3
        let e2 = graph.add_edge(v1, v3, "e2".to_string(), 1.0);
        // v3 -> v1
        let e3 = graph.add_edge(v3, v1, "e3".to_string(), 1.0);
        // v3 -> v2
        let e4 = graph.add_edge(v3, v2, "e4".to_string(), 1.0);
        // v3 -> v4
        let e5 = graph.add_edge(v3, v4, "e5".to_string(), 1.0);
        // v4 -> v1
        let e6 = graph.add_edge(v4, v1, "e6".to_string(), 1.0);
        // v4 -> v2
        let e7 = graph.add_edge(v4, v2, "e7".to_string(), 1.0);

        graph
    }

    fn create_undirected_graph() -> GraphModel<String, f32, f32, petgraph::Undirected, DefaultIx> {
        #![allow(unused_variables)]
        let mut graph = GraphModel::new_undirected();
        let v1 = graph.add_node("v1".to_string(), 1.0);
        let v2 = graph.add_node("v2".to_string(), 1.5);
        let v3 = graph.add_node("v3".to_string(), 1.0);

        // Edges:
        // v1 - v2
        let e1 = graph.add_edge(v1, v2, "e1".to_string(), 1.0);
        // v1 - v3
        let e2 = graph.add_edge(v1, v3, "e2".to_string(), 1.0);
        // v3 - v2
        let e3 = graph.add_edge(v3, v2, "e4".to_string(), 1.0);

        graph
    }

    #[test]
    fn check_bit_map() {
        #![allow(unused_variables)]
        let graph = create_directed_graph();
        let adjacency_list = graph.to_adjacency_list();
        let bit_map = graph.get_bitmap(adjacency_list);
        assert!(
            bit_map.len() == 11,
            "Bit Map has the wrong length! Supposed to be 11 but was {}",
            bit_map.len()
        );
    }

    #[test]
    fn check_serde_with_json() {
        let graph = create_directed_graph();
        let wave = WaveModel::<String, f32, f32>::try_from(graph).unwrap();

        let json_string = serde_json::to_string(&wave).unwrap();
        let from_json_string: WaveModel<String, f32, f32> =
            serde_json::from_str(&json_string).unwrap();
        assert!(wave.is_directed() == from_json_string.is_directed());
    }

    //#[test]
    //fn check_serde_with_toml() {
    //    let graph = create_directed_graph();
    //    let wave = WaveModel::<String, f32, f32>::try_from(graph).unwrap();
    //
    //    let toml_string = toml::to_string(&wave).unwrap();
    //    let from_toml_string: WaveModel<String, f32, f32> = toml::from_str(&toml_string).unwrap();
    //    assert!(wave.is_directed() == from_toml_string.is_directed());
    //}
}
