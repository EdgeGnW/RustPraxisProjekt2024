use super::graphmodel::GraphModel;
use super::QWT;
use qwt::{AccessUnsigned, RankUnsigned, SelectUnsigned, QWT512};
use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeSeq,
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_with;
use std::hash::Hash;
use std::{collections::HashMap, usize};
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
    #[serde_as(as = "Vec<(_, _)>")] //HashMaps are kinda akward to parse. Leave it to some crate
    sequence_index_map: HashMap<L, usize>,
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

fn serialize_bitmap<S>(bitvec_rank9sel: &Rank9Sel, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let bitvec: &BitVector = bitvec_rank9sel.bit_vector();
    let bits: Vec<bool> = bitvec.iter().collect();
    let mut seq = serializer.serialize_seq(Some(bits.len()))?;
    for bit in bits {
        seq.serialize_element(&bit)?;
    }
    seq.end()
}

fn deserialize_bitmap<'de, D>(deserializer: D) -> Result<Rank9Sel, D::Error>
where
    D: Deserializer<'de>,
{
    struct BitVecVisitor;

    impl<'de> Visitor<'de> for BitVecVisitor {
        type Value = Rank9Sel;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a sequence of bools")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Rank9Sel, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut bitvec = BitVector::new();
            while let Some(bit) = seq.next_element::<bool>()? {
                bitvec.push_bit(bit);
            }
            Ok(Rank9Sel::new(bitvec))
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

    fn reconstruct_qwt(&mut self) {
        todo!()
    }

    pub fn rank(&mut self, label: L, n: usize) -> Option<usize> {
        //self.reconstruct_qwt();

        let index = match self.sequence_index_map.get(&label) {
            Some(index) => *index,
            None => return None,
        };

        match self.wavelet_matrix {
            QWT::QWT256(ref qwt) => qwt.rank(index, n),
            QWT::QWT512(ref qwt) => qwt.rank(index, n),
        }
    }

    pub fn access(&mut self, n: usize) -> Option<&(L, N)> {
        //self.reconstruct_qwt();

        match self.wavelet_matrix {
            QWT::QWT256(ref qwt) => {
                let index = match qwt.get(n) {
                    Some(index) => index,
                    None => return None,
                };

                self.data_table_nodes.get(index)
            }
            QWT::QWT512(ref qwt) => {
                let index = match qwt.get(n) {
                    Some(index) => index,
                    None => return None,
                };

                self.data_table_nodes.get(index)
            }
        }
    }

    pub fn select(&mut self, label: L, n: usize) -> Option<usize> {
        //self.reconstruct_qwt();

        let index = match self.sequence_index_map.get(&label) {
            Some(index) => *index,
            None => return None,
        };

        match self.wavelet_matrix {
            QWT::QWT256(ref qwt) => qwt.select(index, n),
            QWT::QWT512(ref qwt) => qwt.select(index, n),
        }
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
                    let label_tupel = match (value.node_label(a), value.node_label(b)) {
                        (Some(a_label), Some(b_label)) => (a_label.clone(), b_label.clone()),
                        _ => return Err(WaveModelError::ConversionError),
                    };

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

        let is_directed = value.is_directed();

        let (data_table_nodes, data_table_edges) = value.into_data_tables();

        //SUPA EXPENSIVE
        let temp_node_index_hash_map: HashMap<&L, usize> = data_table_nodes
            .iter()
            .enumerate()
            .map(|(i, (label, _))| (label, i))
            .collect();

        let sequence_indices = sequence
            .iter()
            .map(|label| temp_node_index_hash_map.get(label).unwrap().clone())
            .collect::<Vec<usize>>();

        let sequence_index_map = sequence
            .iter()
            .cloned()
            .zip(sequence_indices.clone())
            .collect();

        let wavelet_matrix = QWT::QWT256(qwt::QWT256::from(sequence_indices));

        let wavemodel = WaveModel {
            wavelet_matrix,
            sequence,
            sequence_index_map,
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
    use qwt::AccessUnsigned;
    use std::collections::HashMap;
    use sucds::bit_vectors::Rank9Sel;

    fn create_directed_test_model() -> WaveModel<String, usize, usize> {
        let sequence = vec![
            "v2".to_string(),
            "v3".to_string(),
            "v1".to_string(),
            "v1".to_string(),
            "v2".to_string(),
        ];
        let bitmap = Rank9Sel::from_bits([true, false, false, true, false, true, false, false]);
        let mut edge_map = HashMap::<(String, String), Vec<usize>>::new();
        edge_map.insert(("v1".to_string(), "v2".to_string()), vec![0 as usize]);
        edge_map.insert(("v1".to_string(), "v3".to_string()), vec![1 as usize]);
        edge_map.insert(("v2".to_string(), "v1".to_string()), vec![2 as usize]);
        edge_map.insert(("v3".to_string(), "v1".to_string()), vec![3 as usize]);
        edge_map.insert(("v3".to_string(), "v2".to_string()), vec![4 as usize]);

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
        let mut sequence_index_map = HashMap::new();

        for node in &sequence {
            for (n, idx) in &data_table_nodes {
                if node == n {
                    sequence_idx.push(idx.clone());
                    sequence_index_map.insert(n.clone(), idx.clone());
                    break;
                }
            }
        }

        let wavelet_matrix = QWT::QWT256(qwt::QWT256::from(sequence_idx));

        WaveModel {
            wavelet_matrix,
            sequence,
            sequence_index_map,
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
            Rank9Sel::from_bits([true, false, false, true, false, false, true, false, false]);
        let mut edge_map = HashMap::<(String, String), Vec<usize>>::new();
        edge_map.insert(("v1".to_string(), "v2".to_string()), vec![0 as usize]);
        edge_map.insert(("v1".to_string(), "v3".to_string()), vec![1 as usize]);
        edge_map.insert(("v2".to_string(), "v1".to_string()), vec![2 as usize]);
        edge_map.insert(("v2".to_string(), "v3".to_string()), vec![3 as usize]);
        edge_map.insert(("v3".to_string(), "v1".to_string()), vec![4 as usize]);
        edge_map.insert(("v3".to_string(), "v2".to_string()), vec![5 as usize]);

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
        let mut sequence_index_map = HashMap::new();

        for node in &sequence {
            for (n, idx) in &data_table_nodes {
                if node == n {
                    sequence_idx.push(idx.clone());
                    sequence_index_map.insert(n.clone(), idx.clone());
                    break;
                }
            }
        }

        let wavelet_matrix = QWT::QWT256(qwt::QWT256::from(sequence_idx));

        WaveModel {
            wavelet_matrix,
            sequence,
            sequence_index_map,
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

    #[test]
    fn check_sequence_index() {
        let model = create_directed_test_model();
        let expected = vec![1, 2, 0, 0, 1];

        match model.wavelet_matrix() {
            QWT::QWT256(qwt) => {
                for (i, e) in expected.into_iter().enumerate() {
                    assert!(e == qwt.get(i).unwrap())
                }
            }
            QWT::QWT512(qwt) => {
                for (i, e) in expected.into_iter().enumerate() {
                    assert!(e == qwt.get(i).unwrap())
                }
            }
        };
    }

    #[test]
    fn check_rank() {
        let mut model = create_directed_test_model();
        let n = 5; // Kinda weird but n is not inclusive so you have to go one further even
                   // when this value would be larger than the indexing of a Vec
        let data = model.rank("v2".to_owned(), n).unwrap();
        assert!(data == 2);
    }

    #[test]
    fn check_access() {
        let mut model = create_directed_test_model();
        let n = 2;
        let data = model.access(n).unwrap();
        assert!(data.0 == "v1".to_owned());
    }

    #[test]
    fn check_select() {
        let mut model = create_directed_test_model();
        let n = 2;
        let data = model.select("v2".to_owned(), n).unwrap();
        assert!(data == 4);
    }
}
