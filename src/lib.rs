//! # Wavegraph
//! 
//! ## Basic Idea of the Wavegraph Crate
//! 
//! This Crate provides a Datastructure called "Wavegraph" implementing common functionalities of weighted Graphs.
//! To minimize the use of Storage, the Graph will be stored in either one of two different "States" called "Wavemodel" and "Graphmodel".
//! 
//! ### Wavemodel
//! 
//! In the Wavemodel-State, the Graph will be stored in a Wavelet-Matrix Form based on the Crate "[QWT](https://crates.io/crates/qwt)".
//! Some modifying Operations like adding new Nodes is not possible in this State, yet the state is way more efficient in storing the Data then normal Graph implementations
//! based on Adjacency-Lists.
//! 
//! ### Graphmodel
//! 
//! In the Graphmodel-State, the Graph will be stored in an Adjacency-List Form based on the Crate "[Petgraph](https://docs.rs/petgraph/latest/petgraph/)".
//! In this State, most Operations are supported, yet the state is less efficient in storing the Data, then the Wavemodel-State.
 

pub mod graphmodel;
pub mod wavemodel;
pub mod wavegraph;

#[cfg(test)]
mod test {
    use crate::graphmodel::GraphModel;

    #[test]
    fn check_adjacency_list_directed() {
        #![allow(unused_variables)]
        let mut graph = GraphModel::new_directed();
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
        let mut graph = GraphModel::new_undirected();
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
}
