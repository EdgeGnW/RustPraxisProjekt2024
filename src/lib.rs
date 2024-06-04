mod wavegraph;

#[cfg(test)]
mod test {
    use crate::wavegraph::Wavegraph;

    #[test]
    fn check_adjacency_list_directed() {
        #![allow(unused_variables)]
        let mut graph = Wavegraph::new_directed();
        let v1 = graph.add_node(String::from("v1"), 1.0);
        let v2 = graph.add_node(String::from("v2"), 1.5);
        let v3 = graph.add_node(String::from("v3"), 1.0);
        let v4 = graph.add_node(String::from("v4"), 2.0);

        // Edges:
        // v1 -> v2
        let e1 = graph.add_edge(v1.index().into(), v2.index().into(), String::from("e1"), 1.0);
        // v1 -> v3
        let e2 = graph.add_edge(v1.index().into(), v3.index().into(), String::from("e2"), 1.0);
        // v3 -> v1
        let e3 = graph.add_edge(v3.index().into(), v1.index().into(), String::from("e3"), 1.0);
        // v3 -> v2
        let e4 = graph.add_edge(v3.index().into(), v2.index().into(), String::from("e4"), 1.0);
        // v3 -> v4
        let e5 = graph.add_edge(v3.index().into(), v4.index().into(), String::from("e5"), 1.0);
        // v4 -> v1
        let e6 = graph.add_edge(v4.index().into(), v1.index().into(), String::from("e6"), 1.0);
        // v4 -> v2
        let e7 = graph.add_edge(v4.index().into(), v2.index().into(), String::from("e7"), 1.0);

        let adjacency_list = graph.get_adjacency_list();

        assert!(adjacency_list.len() == 4, "Adjacency list has the wrong length!");
        assert!(adjacency_list == vec![vec![&String::from("v2"), &String::from("v3")], vec![], vec![&String::from("v1"), &String::from("v2"), &String::from("v4")], vec![&String::from("v1"), &String::from("v2")]]);
    }

    #[test]
    fn check_adjacency_list_undirected() {
        #![allow(unused_variables)]
        let mut graph = Wavegraph::new_undirected();
        let v1 = graph.add_node(String::from("v1"), 1.0);
        let v2 = graph.add_node(String::from("v2"), 1.5);
        let v3 = graph.add_node(String::from("v3"), 1.0);

        // Edges:
        // v1 - v2
        let e1 = graph.add_edge(v1, v2, String::from("e1"), 1.0);
        // v1 - v3
        let e2 = graph.add_edge(v1, v3, String::from("e2"), 1.0);
        // v3 - v2
        let e4 = graph.add_edge(v3, v2, String::from("e4"), 1.0);

        let adjacency_list = graph.get_adjacency_list();

        assert!(adjacency_list.len() == 3, "Adjacency list has the wrong length!");
        assert!(adjacency_list == vec![vec![&String::from("v2"), &String::from("v3")], vec![&String::from("v1"), &String::from("v3")], vec![&String::from("v1"), &String::from("v2")]]);
    }

    #[test]
    fn check_bit_map() {
        #![allow(unused_variables)]
        let mut graph = Wavegraph::new_directed();
        let v1 = graph.add_node(String::from("v1"), 1.0);
        let v2 = graph.add_node(String::from("v2"), 1.5);
        let v3 = graph.add_node(String::from("v3"), 1.0);
        let v4 = graph.add_node(String::from("v4"), 2.0);

        // Edges:
        // v1 -> v3
        let e1 = graph.add_edge(v1.index().into(), v3.index().into(), String::from("e1"), 1.0);
        // v1 -> v4
        let e2 = graph.add_edge(v1.index().into(), v4.index().into(), String::from("e2"), 1.0);
        // v2 -> v3
        let e3 = graph.add_edge(v2.index().into(), v3.index().into(), String::from("e3"), 1.0);
        // v3 -> v4
        let e4 = graph.add_edge(v3.index().into(), v4.index().into(), String::from("e4"), 1.0);
        // v4 -> v1
        let e6 = graph.add_edge(v4.index().into(), v1.index().into(), String::from("e6"), 1.0);

        let adjacency_list = graph.get_adjacency_list();
        let bit_map = graph.get_bitmap(adjacency_list);
        assert!(bit_map.len() == 9, "Bit Map has the wrong length! Supposed to be 11 but was {}", bit_map.len());
    }
}
