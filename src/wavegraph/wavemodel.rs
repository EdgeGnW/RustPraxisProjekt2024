use super::QWT;

/// # Implementation of the WaveModel-State.
/// In the WaveModel-State, the Graph is stored only in a WaveletMatrix, a BitMap
/// and a flag rather or not the Graph is directed or not.
/// The WaveModel-State has less memory storage usage then the GraphModel-State.
/// Yet because of the way a WaveletMatrix is stored in memory, some operations
/// like changing or deleting Edges or Nodes can't be performed without a change
/// into the GraphModel-State.
pub struct WaveModel<N, E> {
    // wavelet_matrix
    wavelet_matrix: QWT,
    // bit_map
    bit_map: i32, //depends on implementation, only placeholder for now
    // data_table_nodes
    data_table_nodes: Vec<N>,
    // data_table_edges
    data_table_edges: Vec<E>,
    // flag, rather or not the stored Graph is directed
    is_directed: bool,
}
