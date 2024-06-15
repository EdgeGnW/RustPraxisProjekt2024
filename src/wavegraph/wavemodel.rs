use super::graphmodel::GraphModel;
use super::QWT;
use std::collections::HashMap;
use std::hash::Hash;
use sucds::bit_vectors::BitVector;

use petgraph::{
    graph::{DefaultIx, DiGraph, EdgeIndex, IndexType, NodeIndex, UnGraph},
    EdgeType, Graph,
};

#[derive(thiserror::Error, Debug)]
pub enum WaveModelError {
    #[error("THE CONVERSION WENT WRONG")]
    ConversionError,
}
/// # Implementation of the WaveModel-State.
/// In the WaveModel-State, the Graph is stored only in a WaveletMatrix, a BitMap
/// and a flag rather or not the Graph is directed or not.
/// The WaveModel-State has less memory storage usage then the GraphModel-State.
/// Yet because of the way a WaveletMatrix is stored in memory, some operations
/// like changing or deleting Edges or Nodes can't be performed without a change
/// into the GraphModel-State.
pub struct WaveModel<L, N, E> {
    wavelet_matrix: QWT<L>,
    sequence: Vec<L>, //input sequence. A compressed adjacency list. Needs the bitmap to be read.
    bitmap: BitVector, //Here the maximum is (num_of_nodes)²
    edge_map: HashMap<(L, L), usize>, //The key is a tuple of two Node labels
    data_table_nodes: Vec<(L, N)>,
    data_table_edges: Vec<(L, E)>,
    is_directed: bool,
}

impl<L, N, E> WaveModel<L, N, E>
where
    L: Clone + Ord,
{
    pub fn new() -> Self {
        todo!()
    }

    pub fn wavelet_matrix(&self) -> &QWT<L> {
        &self.wavelet_matrix
    }

    pub fn sequence(&self) -> &Vec<L> {
        &self.sequence
    }

    pub fn bitmap(&self) -> &sucds::bit_vectors::BitVector {
        &self.bitmap
    }

    pub fn to_edge_map(&self) -> HashMap<(L, L), usize> {
        self.edge_map.clone()
    }

    //pub fn take_data_table_nodes(&mut self) -> Vec<N> {
    //    mem::take(&mut self.data_table_nodes)
    //}

    //pub fn take_data_table_edges(&mut self) -> Vec<E> {
    //    mem::take(&mut self.data_table_edges)
    //}

    pub fn into_data_tables(self) -> (Vec<(L, N)>, Vec<(L, E)>) {
        (self.data_table_nodes, self.data_table_edges)
    }

    pub fn is_directed(&self) -> bool {
        self.is_directed
    }

    pub fn to_adjacency_list(&self) -> Vec<(L, Vec<L>)> {
        let mut adjacency_list: Vec<(L, Vec<L>)> = Vec::new();
        let mut sequence_iterator = self.sequence().iter();
        let mut current_sequence: Option<Vec<L>> = None;
        let mut node_iterator = self.data_table_nodes.iter();

        for is_node in self.bitmap().iter() {
            if is_node {
                match current_sequence.take() {
                    Some(current_sequence) => {
                        let node_label = node_iterator.next().unwrap().0.clone();
                        adjacency_list.push((node_label, current_sequence));
                    }
                    None => {
                        let node_label = node_iterator.next().unwrap().0.clone();
                        adjacency_list.push((node_label, Vec::new()));
                    }
                }
            } else {
                match &mut current_sequence {
                    None => current_sequence = Some(Vec::new()),
                    Some(current_sequence) => {
                        let label = sequence_iterator.next().unwrap().clone();
                        current_sequence.push(label);
                    }
                }
            }
        }
        adjacency_list
    }
}

impl<L, N, E, Ty, Ix> TryFrom<GraphModel<L, N, E, Ty, Ix>> for WaveModel<L, N, E>
where
    Ty: EdgeType,
    Ix: IndexType,
    L: Clone + Ord + Hash,
{
    type Error = WaveModelError;
    fn try_from(value: GraphModel<L, N, E, Ty, Ix>) -> Result<Self, Self::Error> {
        //TODO: Continue here
        let wavemodel = WaveModel { wavelet_matrix, sequence, bitmap, edge_map, data_table_nodes, data_table_edges, is_directed }
    }
}
