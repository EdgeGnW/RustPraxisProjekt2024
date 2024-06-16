
//! ## Implementation of the WaveModel-State.
//! In the WaveModel-State, the Graph is stored only in a WaveletMatrix, a BitMap
//! and a flag rather or not the Graph is directed or not. Also, the weights of the Nodes
//! and Edges are both stored in seperate Vectors of the matching Datatypes.
//! The WaveModel-State has less memory storage usage then the GraphModel-State.
//! Yet because of the way a WaveletMatrix is stored in memory, some operations
//! like changing or deleting Edges or Nodes can't be performed without a change
//! into the GraphModel-State.


use qwt::{

};


// enum "QWT" should not be in crate "graphmodel", maybe utils-directory?
use crate::graphmodel::QWT;

/// ### The Wavemodel-State of the Graph
/// A Wavemodel consists of a QWT-Waveletmatrix on the Datatyp L for the Labels,
/// a Bitmap represented by a bitvec, Two Vectors Vec<N> and Vec<E> representing
/// the Weights on the Nodes N and the Edges E, aswell as a bool flag indicating
/// if the Graph is directed or not.
/// ```
/// let wavegraph = WaveGraph::from();
/// ```
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