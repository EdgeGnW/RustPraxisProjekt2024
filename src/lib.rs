pub mod wavegraph;
#[cfg(test)]
mod test {
    use crate::wavegraph::GraphModel;
    use crate::wavegraph::WaveModel;
    use std::collections::HashMap;
    use sucds::bit_vectors::BitVector;

    #[test]
    fn check_adjacency_list_directed() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let v1 = graph.add_node("v1", 1.0);
        let v3 = graph.add_node("v3", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v4 = graph.add_node("v4", 2.0);

        // Edges:
        // v1 -> v2
        let e1 = graph.add_edge(v1, v2, "e1", 1.0);
        // v1 -> v3
        let e2 = graph.add_edge(v1, v3, "e2", 1.0);
        // v3 -> v1
        let e3 = graph.add_edge(v3, v1, "e3", 1.0);
        // v3 -> v2
        let e4 = graph.add_edge(v3, v2, "e4", 1.0);
        // v3 -> v4
        let e5 = graph.add_edge(v3, v4, "e5", 1.0);
        // v4 -> v1
        let e6 = graph.add_edge(v4, v1, "e6", 1.0);
        // v4 -> v2
        let e7 = graph.add_edge(v4, v2, "e7", 1.0);

        let adjacency_list = graph.to_adjacency_list();

        assert!(
            adjacency_list.len() == 4,
            "Adjacency list has the wrong length!"
        );
        dbg!(&adjacency_list);
        assert!(
            adjacency_list
                == vec![
                    ("v1", vec!["v2", "v3"]),
                    ("v3", vec!["v1", "v2", "v4"]),
                    ("v2", vec![]),
                    ("v4", vec!["v1", "v2"]),
                ]
        );
    }

    #[test]
    fn check_adjacency_list_undirected() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Undirected> =
            GraphModel::new_undirected();
        let v1 = graph.add_node("v1", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v3 = graph.add_node("v3", 1.0);

        // Edges:
        // v1 - v2
        let e1 = graph.add_edge(v1, v2, "e1", 1.0);
        // v1 - v3
        let e2 = graph.add_edge(v1, v3, "e2", 1.0);
        let e3 = graph.add_edge(v2, v3, "e3", 1.0);
        // v3 - v2
        let e4 = graph.add_edge(v3, v2, "e4", 1.0);

        let adjacency_list = graph.to_adjacency_list();

        assert!(
            adjacency_list.len() == 3,
            "Adjacency list has the wrong length!"
        );

        dbg!(&adjacency_list);

        assert!(
            adjacency_list
                == vec![
                    ("v1", vec!["v2", "v3"]),
                    ("v2", vec!["v1", "v3"]),
                    ("v3", vec!["v1", "v2"]),
                ]
        );
    }

    #[test]
    fn check_transformation_from_graphmodel_directed() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let v1 = graph.add_node("v1", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v3 = graph.add_node("v3", 1.0);

        // Edges:
        // v1 - v2
        let e1 = graph.add_edge(v1, v2, "e1", 1.0);
        // v1 - v3
        let e2 = graph.add_edge(v1, v3, "e2", 1.0);
        // v2 - v3
        let e3 = graph.add_edge(v2, v3, "e3", 1.0);
        // v3 - v2
        let e4 = graph.add_edge(v3, v2, "e4", 1.0);

        match WaveModel::try_from(graph) {
            Ok(wavemodel) => {
                // sequence
                let sequence_exp = vec!["v2", "v3", "v3", "v2"];
                let sequence_found = wavemodel.sequence().clone();
                assert!(sequence_exp == sequence_found);

                // bitmap
                let bitmap_exp =
                    BitVector::from_bits([true, false, false, true, false, true, false]);
                let bitmap_found = wavemodel.bitmap().clone();
                assert!(bitmap_exp == bitmap_found);

                // is_directed
                assert!(wavemodel.is_directed());

                // edge_map
                let mut edge_map_expected = HashMap::new();
                edge_map_expected.insert(("v1", "v2"), 0); // e1
                edge_map_expected.insert(("v1", "v3"), 1); // e2
                edge_map_expected.insert(("v2", "v3"), 2); // e3
                edge_map_expected.insert(("v3", "v2"), 3); // e4
                let edge_map_found = wavemodel.to_edge_map();
                assert!(edge_map_found == edge_map_expected);
            }
            Err(e) => {
                dbg!(e);
                assert!(false);
            }
        }
    }

    #[test]
    fn check_transformation_from_graphmodel_undirected() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Undirected> =
            GraphModel::new_undirected();
        let v1 = graph.add_node("v1", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v3 = graph.add_node("v3", 1.0);

        // Edges:
        // v1 - v2
        let e1 = graph.add_edge(v1, v2, "e1", 1.0);
        // v1 - v3
        let e2 = graph.add_edge(v1, v3, "e2", 1.0);
        // v2 - v3
        let e3 = graph.add_edge(v2, v3, "e3", 1.0);
        // v3 - v2
        let e4 = graph.add_edge(v3, v2, "e4", 1.0);

        match WaveModel::try_from(graph) {
            Ok(wavemodel) => {
                // sequence
                let sequence_exp = vec!["v2", "v3", "v1", "v3", "v1", "v2"];
                let sequence_found = wavemodel.sequence().clone();
                assert!(sequence_exp == sequence_found);

                // bitmap
                let bitmap_exp = BitVector::from_bits([
                    true, false, false, true, false, false, true, false, false,
                ]);
                let bitmap_found = wavemodel.bitmap().clone();
                assert!(bitmap_exp == bitmap_found);

                // is_directed
                assert!(!wavemodel.is_directed());

                // edge_map
                let mut edge_map_expected = HashMap::new();
                edge_map_expected.insert(("v1", "v2"), 0); // e1
                edge_map_expected.insert(("v1", "v3"), 1); // e2
                edge_map_expected.insert(("v2", "v3"), 2); // e3
                edge_map_expected.insert(("v3", "v2"), 3); // e4
                let edge_map_found = wavemodel.to_edge_map();
                assert!(edge_map_found == edge_map_expected);
            }
            Err(e) => {
                dbg!(e);
                assert!(false);
            }
        }
    }

    #[test]
    fn check_transformation_from_wavemodel_directed() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let v1 = graph.add_node("v1", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v3 = graph.add_node("v3", 1.0);

        // Edges:
        // v1 - v2
        let e1 = graph.add_edge(v1, v2, "e1", 1.0);
        // v1 - v3
        let e2 = graph.add_edge(v1, v3, "e2", 1.0);
        // v2 - v3
        let e3 = graph.add_edge(v2, v3, "e3", 1.0);
        // v3 - v2
        let e4 = graph.add_edge(v3, v2, "e4", 1.0);

        let mut graph_orig = graph.clone();

        let wavemodel: WaveModel<&str, f64, f64>;

        match WaveModel::try_from(graph) {
            Ok(w) => {
                wavemodel = w;
            }
            Err(e) => {
                wavemodel = WaveModel::new();
                dbg!(e);
                assert!(false);
            }
        }

        match GraphModel::<&str, f64, f64, petgraph::prelude::Directed>::try_from(wavemodel) {
            Ok(graphmodel) => {
                let adjacency_list_expected = vec![
                    ("v1", vec!["v2", "v3"]),
                    ("v2", vec!["v3"]),
                    ("v3", vec!["v2"]),
                ];
                let adjacency_list_found = graphmodel.to_adjacency_list();

                dbg!(&adjacency_list_expected);
                dbg!(&adjacency_list_found);
                assert!(adjacency_list_found == adjacency_list_expected);
            }
            Err(e) => {}
        }
    }
}
