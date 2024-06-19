pub mod wavegraph;
#[cfg(test)]
mod test {
    use crate::wavegraph::GraphModel;
    use crate::wavegraph::WaveModel;
    use std::collections::HashMap;
    use sucds::bit_vectors::BitVector;
    use petgraph::graph::DefaultIx;

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
            vec![&"v2", &"v3"],
            vec![],
            vec![&"v1", &"v2", &"v4"],
            vec![&"v1", &"v2"],
        ];
        assert!(
            adjacency_list == expected_list,
            "Adjacency list diverges from expectation.\nExpected\n{0:?}\nFound:\n{1:?}",
            expected_list,
            adjacency_list
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
                assert!(false);
            }
      }
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

        let expected_list = vec![vec![&"v2", &"v3"], vec![&"v1", &"v3"], vec![&"v1", &"v2"]];
        assert!(
            adjacency_list == expected_list,
            "Adjacency list diverges from expectation.\nExpected:\n{0:?}\nFound:\n{1:?}",
            expected_list,
            adjacency_list
        );
    }

    #[test]
    fn check_update_node_label() {
        #![allow(unused_variables)]
        let mut graph_dir = create_directed_graph();
        let mut graph_undir = create_undirected_graph();

        for i in 0..3 {
            let label_old = graph_dir.node_label(i.into()).unwrap().clone();
            graph_dir.update_node_label(i.into(), format!("w{0}", i).to_string());
            let label_new = graph_dir.node_label(i.into()).unwrap().clone();
            assert!(
                label_old != label_new,
                "[DirGraph] Label was not changed by the update!\nOld:\n{0}\nNew:\n{1}",
                label_old,
                label_new
            );
            assert!(
                format!("w{0}", i).to_string() == label_new,
                "[DirGraph] Label diverges from expected value!\nExpected:\nw{0}\nFound:\n{1}",
                i,
                label_new
            );
        }

        for i in 0..2 {
            let label_old = graph_undir.node_label(i.into()).unwrap().clone();
            graph_undir.update_node_label(i.into(), format!("w{0}", i).to_string());
            let label_new = graph_undir.node_label(i.into()).unwrap().clone();
            assert!(
                label_old != label_new,
                "[UndirGraph] Label was not changed by the update!\nOld:\n{0}\nNew:\n{1}",
                label_old,
                label_new
            );
            assert!(
                format!("w{0}", i).to_string() == label_new,
                "[UndirGraph] Label diverges from expected value!\nExpected:\nw{0}\nFound:\n{1}",
                i,
                label_new
            );
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

        let adjacency_list_expected = vec![
            ("v1", vec!["v2", "v3"]),
            ("v2", vec!["v3"]),
            ("v3", vec!["v2"]),
        ];

        assert!(adjacency_list_expected == graph.to_adjacency_list());

        let mut graph_orig = graph.clone();

        let wavemodel: WaveModel<&str, f64, f64>;

        match WaveModel::try_from(graph) {
            Ok(w) => {
                wavemodel = w;
            }
            Err(e) => {
                wavemodel = WaveModel::new();
                assert!(false);
            }
        }

        match GraphModel::<&str, f64, f64, petgraph::prelude::Directed>::try_from(wavemodel) {
            Ok(graphmodel) => {
                let adjacency_list_found = graphmodel.to_adjacency_list();
                dbg!(&adjacency_list_found);
                dbg!(&adjacency_list_expected);

                assert!(adjacency_list_found == adjacency_list_expected);
            }
            Err(e) => {}
        }
    }

    fn check_update_edge_label() {
        #![allow(unused_variables)]
        let mut graph_dir = create_directed_graph();
        let mut graph_undir = create_undirected_graph();

        for i in 0..6 {
            let label_old = graph_dir.edge_label(i.into()).unwrap().clone();
            graph_dir.update_edge_label(i.into(), format!("w{0}", i).to_string());
            let label_new = graph_dir.edge_label(i.into()).unwrap().clone();
            assert!(
                label_old != label_new,
                "[DirGraph] Label was not changed by the update!\nOld:\n{0}\nNew:\n{1}",
                label_old,
                label_new
            );
            assert!(
                format!("w{0}", i).to_string() == label_new,
                "[DirGraph] Label diverges from expected value!\nExpected:\nw{0}\nFound:\n{1}",
                i,
                label_new
            );
        }

        for i in 0..2 {
            let label_old = graph_undir.edge_label(i.into()).unwrap().clone();
            graph_undir.update_edge_label(i.into(), format!("w{0}", i).to_string());
            let label_new = graph_undir.edge_label(i.into()).unwrap().clone();
            assert!(
                label_old != label_new,
                "[UndirGraph] Label was not changed by the update!\nOld:\n{0}\nNew:\n{1}",
                label_old,
                label_new
            );
            assert!(
                format!("w{0}", i).to_string() == label_new,
                "[UndirGraph] Label diverges from expected value!\nExpected:\nw{0}\nFound:\n{1}",
                i,
                label_new
            );
        }
    }

    #[test]
    fn check_bit_map() {
        #![allow(unused_variables)]
        let mut graph = create_directed_graph();
        let adjacency_list = graph.to_adjacency_list();
        let bit_map = graph.get_bitmap(adjacency_list);
        assert!(
            bit_map.len() == 11,
            "Bit Map has the wrong length! Supposed to be 11 but was {}",
            bit_map.len()
        );
    }
}
