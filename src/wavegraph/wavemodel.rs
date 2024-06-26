use super::graphmodel::GraphModel;
use super::QWT;
use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeSeq,
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_with;
use std::collections::HashMap;
use std::hash::Hash;
use sucds::bit_vectors::{BitVector, Rank9Sel};

use petgraph::{graph::IndexType, EdgeType};

#[derive(thiserror::Error, Debug)]
pub enum WaveModelError {
    #[error("THE CONVERSION WENT WRONG")]
    ConversionError,
    #[error("INVALID EDGE. NO ENDPOINTS FOUND")]
    InvalidEdge,
}

/// # Implementation of the WaveModel-State.
/// In the WaveModel-State, the Graph is stored only in a WaveletMatrix, a BitMap
/// and a flag rather or not the Graph is directed or not.
/// The WaveModel-State has less memory storage usage then the GraphModel-State.
/// Yet because of the way a WaveletMatrix is stored in memory, some operations
/// like changing or deleting Edges or Nodes can't be performed without a change
/// into the GraphModel-State.
#[serde_with::serde_as]
#[derive(Clone, Serialize, Deserialize)]
pub struct WaveModel<L, N, E>
where
    L: Ord + Hash, //We need this trait bound for the serialization to work
{
    wavelet_matrix: QWT,
    sequence: Vec<L>, //input sequence. A compressed adjacency list. Needs the bitmap to be read.
    #[serde(
        serialize_with = "serialize_bitmap",
        deserialize_with = "deserialize_bitmap"
    )]
    bitmap: Rank9Sel, //Here the maximum is (num_of_nodes)²
    #[serde_as(as = "Vec<(_, _)>")] //HashMaps are kinda akward to parse. Leave it to some crate
    edge_map: HashMap<(L, L), Vec<usize>>, //The key is a tuple of two Node labels
    data_table_nodes: Vec<(L, N)>,
    data_table_edges: Vec<(L, E)>,
    is_directed: bool,
}

fn serialize_bitmap<S>(bitvec: &BitVector, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let bits: Vec<bool> = bitvec.iter().collect();
    let mut seq = serializer.serialize_seq(Some(bits.len()))?;
    for bit in bits {
        seq.serialize_element(&bit)?;
    }
    seq.end()
}

fn deserialize_bitmap<'de, D>(deserializer: D) -> Result<BitVector, D::Error>
where
    D: Deserializer<'de>,
{
    struct BitVecVisitor;

    impl<'de> Visitor<'de> for BitVecVisitor {
        type Value = BitVector;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a sequence of bools")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<BitVector, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut bitvec = BitVector::new();
            while let Some(bit) = seq.next_element::<bool>()? {
                bitvec.push_bit(bit);
            }
            Ok(bitvec)
        }
    }

    deserializer.deserialize_seq(BitVecVisitor)
}

impl<L, N, E> WaveModel<L, N, E>
where
    L: Clone + Ord + Hash,
{
    pub fn new() -> Self {
        todo!()
    }

    pub fn wavelet_matrix(&self) -> &QWT {
        &self.wavelet_matrix
    }

    pub fn sequence(&self) -> &Vec<L> {
        &self.sequence
    }

    pub fn bitmap(&self) -> &Rank9Sel {
        &self.bitmap
    }

    pub fn to_edge_map(&self) -> HashMap<(L, L), Vec<usize>> {
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
        let mut current_sequence: Vec<L> = Vec::new();
        let mut node_iterator = self.data_table_nodes.iter();
        let mut bitmap_iter = self.bitmap().bit_vector().iter().peekable();

        while let Some(is_node) = bitmap_iter.next() {
            if !is_node {
                let label = sequence_iterator.next().unwrap().clone();
                current_sequence.push(label);
            }

            // If the next bit indicates the beginning of a new node..
            if let Some(is_node) = bitmap_iter.peek() {
                if *is_node {
                    let node_label = node_iterator.next().unwrap().0.clone();
                    // ..add the current sequence to the adjacency list
                    adjacency_list.push((node_label, current_sequence.clone()));
                    current_sequence = Vec::new();
                }
            }
        }

        adjacency_list.push((
            node_iterator.next().unwrap().0.clone(),
            current_sequence.clone(),
        ));

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
        let adjacency_list = value.to_adjacency_list();
        let mut edge_map: HashMap<(L, L), Vec<usize>> = HashMap::new();

        let mut bitmap_vec = BitVector::new();

        let mut sequence = Vec::new();
        for (_, neighbor_labels) in adjacency_list {
            bitmap_vec.push_bit(true);
            for neighbor_label in neighbor_labels {
                bitmap_vec.push_bit(false);
                sequence.push(neighbor_label);
            }
        }

        let bitmap = Rank9Sel::new(bitmap_vec);

        for edge_index in value.edge_indicies() {
            match value.edge_endpoints(edge_index) {
                Some((a, b)) => {
                    let a_label = match value.node_label(a) {
                        Some(label) => label.clone(),
                        None => return Err(WaveModelError::ConversionError),
                    };

                    let b_label = match value.node_label(b) {
                        Some(label) => label.clone(),
                        None => return Err(WaveModelError::ConversionError),
                    };

                    let label_tupel = (a_label, b_label);
                    let possible_edge = edge_map.get_mut(&label_tupel);
                    match possible_edge {
                        Some(edge) => edge.push(edge_index.index()),
                        None => {
                            edge_map.insert(label_tupel, Vec::from([edge_index.index()]));
                        }
                    }
                }
                None => return Err(WaveModelError::InvalidEdge),
            }
        }

        let sequence_indices = (0..sequence.len()).collect::<Vec<usize>>();

        let wavelet_matrix = QWT::QWT256(qwt::QWT256::from(sequence_indices));
        let is_directed = value.is_directed();

        let (data_table_nodes, data_table_edges) = value.into_data_tables();

        let wavemodel = WaveModel {
            wavelet_matrix,
            sequence,
            bitmap,
            edge_map,
            data_table_nodes,
            data_table_edges,
            is_directed,
        };

        Ok(wavemodel)
    }
}

#[cfg(test)]
mod test {
    use super::QWT;
    use crate::wavegraph::WaveModel;
    use std::collections::HashMap;
    use sucds::bit_vectors::BitVector;

    fn create_directed_test_model() -> WaveModel<String, usize, usize> {
        let sequence = vec![
            "v2".to_string(),
            "v3".to_string(),
            "v1".to_string(),
            "v1".to_string(),
            "v2".to_string(),
        ];
        let bitmap = BitVector::from_bits([true, false, false, true, false, true, false, false]);
        let mut edge_map = HashMap::<(String, String), usize>::new();
        edge_map.insert(("v1".to_string(), "v2".to_string()), 0);
        edge_map.insert(("v1".to_string(), "v3".to_string()), 1);
        edge_map.insert(("v2".to_string(), "v1".to_string()), 2);
        edge_map.insert(("v3".to_string(), "v1".to_string()), 3);
        edge_map.insert(("v3".to_string(), "v2".to_string()), 4);

        // The indexing of the nodes may be wrong
        let data_table_nodes = vec![
            ("v1".to_string(), 0),
            ("v2".to_string(), 1),
            ("v3".to_string(), 2),
        ];
        let data_table_edges = vec![
            ("e1".to_string(), 0),
            ("e2".to_string(), 1),
            ("e3".to_string(), 2),
            ("e4".to_string(), 3),
            ("e5".to_string(), 4),
        ];

        let mut sequence_idx: Vec<usize> = Vec::new();

        for node in &sequence {
            for (n, idx) in &data_table_nodes {
                if node == n {
                    sequence_idx.push(idx.clone());
                    break;
                }
            }
        }

        let wavelet_matrix = QWT::QWT256(qwt::QWT256::from(sequence_idx));

        WaveModel {
            wavelet_matrix,
            sequence,
            bitmap,
            edge_map,
            data_table_nodes,
            data_table_edges,
            is_directed: true,
        }
    }

    fn create_undirected_test_model() -> WaveModel<String, usize, usize> {
        let sequence = vec![
            "v2".to_string(),
            "v3".to_string(),
            "v1".to_string(),
            "v2".to_string(),
            "v1".to_string(),
            "v2".to_string(),
        ];
        let bitmap =
            BitVector::from_bits([true, false, false, true, false, false, true, false, false]);
        let mut edge_map = HashMap::<(String, String), usize>::new();
        edge_map.insert(("v1".to_string(), "v2".to_string()), 0);
        edge_map.insert(("v1".to_string(), "v3".to_string()), 1);
        edge_map.insert(("v2".to_string(), "v1".to_string()), 2);
        edge_map.insert(("v2".to_string(), "v3".to_string()), 3);
        edge_map.insert(("v3".to_string(), "v1".to_string()), 4);
        edge_map.insert(("v3".to_string(), "v2".to_string()), 5);

        // The indexing of the nodes may be wrong
        let data_table_nodes = vec![
            ("v1".to_string(), 0),
            ("v2".to_string(), 1),
            ("v3".to_string(), 2),
        ];
        let data_table_edges = vec![
            ("e1".to_string(), 0),
            ("e2".to_string(), 1),
            ("e3".to_string(), 2),
            ("e4".to_string(), 3),
            ("e5".to_string(), 4),
            ("e6".to_string(), 5),
        ];

        let mut sequence_idx: Vec<usize> = Vec::new();

        for node in &sequence {
            for (n, idx) in &data_table_nodes {
                if node == n {
                    sequence_idx.push(idx.clone());
                    break;
                }
            }
        }

        let wavelet_matrix = QWT::QWT256(qwt::QWT256::from(sequence_idx));

        WaveModel {
            wavelet_matrix,
            sequence,
            bitmap,
            edge_map,
            data_table_nodes,
            data_table_edges,
            is_directed: false,
        }
    }

    #[test]
    fn check_adjacency_list_directed() {
        let model = create_directed_test_model();
        let found = model.to_adjacency_list();

        let expected = vec![
            ("v1".to_string(), vec!["v2".to_string(), "v3".to_string()]),
            ("v2".to_string(), vec!["v1".to_string()]),
            ("v3".to_string(), vec!["v1".to_string(), "v2".to_string()]),
        ];

        assert!(
            found == expected,
            "Adjacency list was not as expected!\nExpected: {0:?}\nFound: {1:?}",
            expected,
            found
        );
    }

    fn check_adjacency_list_undirected() {
        let model = create_undirected_test_model();
        let found = model.to_adjacency_list();

        let expected = vec![
            ("v1".to_string(), vec!["v2".to_string(), "v3".to_string()]),
            ("v2".to_string(), vec!["v1".to_string(), "v3".to_string()]),
            ("v3".to_string(), vec!["v1".to_string(), "v2".to_string()]),
        ];

        assert!(
            found == expected,
            "Adjacency list was not as expected!\nExpected: {0:?}\nFound: {1:?}",
            expected,
            found
        );
    }
}
