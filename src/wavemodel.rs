use std::convert::From;

use qwt::{
    QWT256,
    QWT512,
};


// enum "QWT" should not be in crate "graphmodel", maybe utils-directory?
use crate::graphmodel::QWT;
use crate::graphmodel::GraphModel;

/// # Implementation of the WaveModel-State.
/// In the WaveModel-State, the Graph is stored only in a WaveletMatrix, a BitMap
/// and a flag rather or not the Graph is directed or not.
/// The WaveModel-State has less memory storage usage then the GraphModel-State.
/// Yet because of the way a WaveletMatrix is stored in memory, some operations
/// like changing or deleting Edges or Nodes can't be performed without a change
/// into the GraphModel-State.
pub struct WaveModel<L,N,E>{
    // wavelet_matrix
    wavelet_matrix: QWT<L>,
    // bit_map 
    bit_map: i32, //depends on implementation, only placeholder for now
    // data_table_nodes
    data_table_nodes: Vec<N>,
    // data_table_edges
    data_table_edges: Vec<E>,
    // flag, rather or not the stored Graph is directed
    is_directed: bool,
}

impl<L, N, E> WaveModel<L, N, E> {
    pub fn new() -> Self {
        todo!()
    }
}

impl<L, N, E, Ix> From<GraphModel<L, N, E, petgraph::Directed, Ix>> for WaveModel<L, N, E> 
where
    Ix: petgraph::adj::IndexType,
    L: Clone,
{
    fn from(graph: GraphModel<L, N, E, petgraph::Directed, Ix>) -> Self {
        // Somehow decide whether to take QWT256 or QWT512
        let adj_list = graph.to_adjacency_list().iter().flatten().collect::<Vec<_>>();

        // Get indices for each label and replace within adj_list

        WaveModel {
            wavelet_matrix: todo!(),
            bit_map: todo!(),
            data_table_nodes: todo!(),
            data_table_edges: todo!(),
            is_directed: todo!(),
        }
    }
}

impl<L, N, E, Ix> From<GraphModel<L, N, E, petgraph::Undirected, Ix>> for WaveModel<L, N, E> 
where
    Ix: petgraph::adj::IndexType,
    L: Clone,
{
    fn from(graph: GraphModel<L, N, E, petgraph::Undirected, Ix>) -> Self {
        WaveModel {
            wavelet_matrix: todo!(),
            bit_map: todo!(),
            data_table_nodes: todo!(),
            data_table_edges: todo!(),
            is_directed: todo!(),
        }
    }
}
