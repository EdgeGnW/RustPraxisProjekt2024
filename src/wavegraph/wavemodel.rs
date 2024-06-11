use std::collections::HashMap;

use super::graphmodel::GraphModel;
use super::QWT;

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
    bitmap: usize,    //Here the maximum is (num_of_nodes)²
    edge_map: HashMap<(L, L), usize>, //The key is a tuple of two Node labels
    //TODO: Change the type of the hashmap value and the bitmap to only take up as much space as it needs
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

    pub fn bitmap(&self) -> usize {
        self.bitmap
    }

    pub fn edge_map(&self) -> &HashMap<(L, L), usize> {
        &self.edge_map
    }

    //pub fn take_data_table_nodes(&mut self) -> Vec<N> {
    //    mem::take(&mut self.data_table_nodes)
    //}

    //pub fn take_data_table_edges(&mut self) -> Vec<E> {
    //    mem::take(&mut self.data_table_edges)
    //}

    pub fn data_tables(self) -> (Vec<(L, N)>, Vec<(L, E)>) {
        (self.data_table_nodes, self.data_table_edges)
    }

    pub fn is_directed(&self) -> bool {
        self.is_directed
    }
}
