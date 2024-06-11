pub mod wavegraph;
#[cfg(test)]
mod test {
    use crate::wavegraph::GraphModel;

    #[test]
    fn check_adjacency_list_directed() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let v1 = graph.add_node("v1", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v3 = graph.add_node("v3", 1.0);
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
                    ("v2", vec![]),
                    ("v3", vec!["v1", "v2", "v4"]),
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
}
