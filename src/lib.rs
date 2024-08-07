pub mod wavegraph;
#[cfg(test)]
mod test {
    use crate::wavegraph::GraphModel;
    use crate::wavegraph::WaveModel;
    use petgraph::graph::DefaultIx;
    use std::collections::HashMap;
    use sucds::bit_vectors::Rank9Sel;

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
    fn check_adjacency_list_directed() {
        let graph = create_directed_graph();

        let adjacency_list = graph.to_adjacency_list();

        let expected_len = 4;
        assert!(
            adjacency_list.len() == expected_len,
            "Adjacency list has the wrong length!\nExpected:\n{0:?}\nFound:\n{1}",
            expected_len,
            adjacency_list.len()
        );

        let expected_list = vec![
            ("v1".to_string(), vec!["v2".to_string(), "v3".to_string()]),
            ("v2".to_string(), vec![]),
            (
                "v3".to_string(),
                vec!["v1".to_string(), "v2".to_string(), "v4".to_string()],
            ),
            ("v4".to_string(), vec!["v1".to_string(), "v2".to_string()]),
        ];
        assert!(
            adjacency_list == expected_list,
            "Adjacency list diverges from expectation.\nExpected\n{0:?}\nFound:\n{1:?}",
            expected_list,
            adjacency_list
        );
    }

    fn check_adjacency_list_undirected() {
        let graph = create_undirected_graph();

        let adjacency_list = graph.to_adjacency_list();

        let expected_len = 3;
        assert!(
            adjacency_list.len() == expected_len,
            "Adjacency list has the wrong length!\nExpected:\n{0:?}\nFound:\n{1}",
            expected_len,
            adjacency_list.len()
        );

        let expected_list = vec![
            ("v1".to_string(), vec!["v2".to_string(), "v3".to_string()]),
            ("v2".to_string(), vec!["v1".to_string(), "v3".to_string()]),
            ("v3".to_string(), vec!["v1".to_string(), "v2".to_string()]),
        ];
        assert!(
            adjacency_list == expected_list,
            "Adjacency list diverges from expectation.\nExpected:\n{0:?}\nFound:\n{1:?}",
            expected_list,
            adjacency_list
        );
    }

    #[test]
    fn check_transformation_from_graphmodel_directed() {
        #![allow(unused_variables)]
        let graph: GraphModel<String, f32, f32, petgraph::prelude::Directed> =
            create_directed_graph();

        match WaveModel::try_from(graph) {
            Ok(wavemodel) => {
                // sequence
                let sequence_exp = vec![
                    "v2".to_string(),
                    "v3".to_string(),
                    "v1".to_string(),
                    "v2".to_string(),
                    "v4".to_string(),
                    "v1".to_string(),
                    "v2".to_string(),
                ];
                let sequence_found = wavemodel.sequence().clone();
                assert!(
                    sequence_exp == sequence_found,
                    "Sequence not as expected!\nExpected: {0:?}\nFound: {1:?}",
                    sequence_exp,
                    sequence_found
                );

                // bitmap
                let bitmap_exp = Rank9Sel::from_bits([
                    true, false, false, true, true, false, false, false, true, false, false,
                ]);
                let bitmap_found = wavemodel.bitmap().clone();
                assert!(
                    bitmap_exp == bitmap_found,
                    "Bitmap not as expected!\nExpected: {0:?}\nFound: {1:?}",
                    bitmap_exp,
                    bitmap_found
                );

                // is_directed
                assert!(wavemodel.is_directed(), "WaveModel is not directed!");

                // edge_map
                let mut edge_map_expected = HashMap::new();
                edge_map_expected.insert(("v1".to_string(), "v2".to_string()), vec![0 as usize]); // e1
                edge_map_expected.insert(("v1".to_string(), "v3".to_string()), vec![1 as usize]); // e2
                edge_map_expected.insert(("v3".to_string(), "v1".to_string()), vec![2 as usize]); // e3
                edge_map_expected.insert(("v3".to_string(), "v2".to_string()), vec![3 as usize]); // e4
                edge_map_expected.insert(("v3".to_string(), "v4".to_string()), vec![4 as usize]); // e5
                edge_map_expected.insert(("v4".to_string(), "v1".to_string()), vec![5 as usize]); // e6
                edge_map_expected.insert(("v4".to_string(), "v2".to_string()), vec![6 as usize]); // e7
                let edge_map_found = wavemodel.to_edge_map();
                assert!(
                    edge_map_found == edge_map_expected,
                    "Edge map not as expected!\nExpected: {0:?}\nFound: {1:?}",
                    edge_map_expected,
                    edge_map_found
                );
            }
            Err(e) => {
                assert!(false, "Transformation failed!");
            }
        }
    }

    #[test]
    fn check_transformation_from_graphmodel_undirected() {
        #![allow(unused_variables)]
        let graph: GraphModel<String, f32, f32, petgraph::prelude::Undirected> =
            create_undirected_graph();

        match WaveModel::try_from(graph) {
            Ok(wavemodel) => {
                // sequence
                let sequence_exp = vec![
                    "v2".to_string(),
                    "v3".to_string(),
                    "v1".to_string(),
                    "v3".to_string(),
                    "v1".to_string(),
                    "v2".to_string(),
                ];
                let sequence_found = wavemodel.sequence().clone();
                assert!(
                    sequence_exp == sequence_found,
                    "Sequence not as expected!\nExpected: {0:?}\nFound: {1:?}",
                    sequence_exp,
                    sequence_found
                );

                // bitmap
                let bitmap_exp = Rank9Sel::from_bits([
                    true, false, false, true, false, false, true, false, false,
                ]);
                let bitmap_found = wavemodel.bitmap().clone();
                assert!(
                    bitmap_exp == bitmap_found,
                    "Bitmap not as expected!\nExpected: {0:?}\nFound: {1:?}",
                    bitmap_exp,
                    bitmap_found
                );

                // is_directed
                assert!(!wavemodel.is_directed(), "WaveModel is not directed!");

                // edge_map
                let mut edge_map_expected = HashMap::new();
                edge_map_expected.insert(("v1".to_string(), "v2".to_string()), vec![0 as usize]); // e1
                edge_map_expected.insert(("v1".to_string(), "v3".to_string()), vec![1 as usize]); // e2
                edge_map_expected.insert(("v3".to_string(), "v2".to_string()), vec![2 as usize]); // e3
                let edge_map_found = wavemodel.to_edge_map();
                assert!(
                    edge_map_found == edge_map_expected,
                    "Edge map not as expected!\nExpected: {0:?}\nFound: {1:?}",
                    edge_map_expected,
                    edge_map_found
                );
            }
            Err(e) => {
                assert!(false, "Transformation failed!");
            }
        }
    }

    #[test]
    fn check_transformation_from_wavemodel_directed() {
        #![allow(unused_variables)]
        let graph: GraphModel<String, f32, f32, petgraph::prelude::Directed> =
            create_directed_graph();

        let graph_orig = graph.clone();

        let wavemodel: WaveModel<String, f32, f32>;

        match WaveModel::try_from(graph) {
            Ok(w) => {
                wavemodel = w;
            }
            Err(e) => {
                wavemodel = WaveModel::new();
                assert!(false);
            }
        }

        match GraphModel::<String, f32, f32, petgraph::prelude::Directed>::try_from(wavemodel) {
            Ok(graphmodel) => {
                let adjacency_list_expected = vec![
                    ("v1".to_string(), vec!["v2".to_string(), "v3".to_string()]),
                    ("v2".to_string(), vec![]),
                    (
                        "v3".to_string(),
                        vec!["v1".to_string(), "v2".to_string(), "v4".to_string()],
                    ),
                    ("v4".to_string(), vec!["v1".to_string(), "v2".to_string()]),
                ];

                let adjacency_list_found = graphmodel.to_adjacency_list();

                assert!(
                    adjacency_list_found == adjacency_list_expected,
                    "Adjacency list not as expected!\nExpected: {0:?}\nFound: {1:?}",
                    adjacency_list_expected,
                    adjacency_list_found
                );

                assert!(graphmodel.is_directed(), "GraphModel is not directed!");

                let bitmap_exp = Rank9Sel::from_bits([
                    true, false, false, true, true, false, false, false, true, false, false,
                ]);
                let bitmap_found = graphmodel.to_bitmap(adjacency_list_found);

                assert!(
                    bitmap_found.bit_vector() == bitmap_exp.bit_vector(),
                    "Bitmap not as expected!\nExpected: {0:?}\nFound: {1:?}",
                    bitmap_exp,
                    bitmap_found
                );
            }
            Err(e) => {
                assert!(false, "Transformation failed!");
            }
        }
    }

    #[test]
    fn check_bit_map() {
        #![allow(unused_variables)]
        let graph = create_directed_graph();
        let adjacency_list = graph.to_adjacency_list();
        let bit_map = graph.to_bitmap(adjacency_list);
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
