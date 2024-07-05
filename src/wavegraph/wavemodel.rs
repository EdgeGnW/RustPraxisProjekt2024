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
    #[error("NODE DOES NOT EXIST")]
    NodeDoesNotExist,
    #[error("EDGE DOES NOT EXIST")]
    EdgeDoesNotExist,
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
    #[serde(skip)]
    is_modified: bool,
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

    pub fn is_modified(&self) -> bool {
        self.is_modified
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

    /// Adds a new node to the WaveModel. Returns the index of the newly added node.
    pub fn add_node(&mut self, label: L, weight: N) -> usize {
        self.data_table_nodes.push((label, weight));
        return self.data_table_nodes.len() - 1;
    }

    /// Allows updating the label and/or weight of a node. Note that the tuple containing the label
    /// and weight allows for both to be `None`. In the case of either being `None`, no operation
    /// is performed on the particular data referred to the node.
    /// Only ever returns an error if the passed index is out of bounds. In every other case, even
    /// when no data is changed, Ok(idx) is returned.
    pub fn update_node(
        &mut self,
        idx: usize,
        (new_label, new_weight): (Option<L>, Option<N>),
    ) -> Result<usize, WaveModelError> {
        if let Some((label, weight)) = self.data_table_nodes.get_mut(idx) {
            if let Some(new_l) = new_label {
                *label = new_l;
            }
            if let Some(new_w) = new_weight {
                *weight = new_w;
            }
            Ok(idx)
        } else {
            Err(WaveModelError::NodeDoesNotExist)
        }
    }

    /// Removes a node from the WaveModel. Returns an error if the passed index is out of bounds.
    /// Otherwise returns the removed node as (label_node, weight_node)-tupel.
    /// Note that calling this function will also remove edges containing the to-be-removed-node.
    pub fn remove_node(&mut self, idx: usize) -> Result<(L, N), WaveModelError> {
        let node_label_opt: Option<L>;
        if let Some(label) = self.node_label(&idx) {
            // Can be safely unwrapped after this
            node_label_opt = Some(label.clone());
        } else {
            return Err(WaveModelError::NodeDoesNotExist);
        }

        let node_label = node_label_opt.unwrap();

        // Remove node from edge_map
        let mut keys_remove: Vec<(L, L)> = Vec::new();
        for (key, _) in &self.edge_map {
            if key.0 == node_label || key.1 == node_label {
                // Remember vector of keys to delete later
                keys_remove.push(key.clone());
            }
        }

        // Remove keys from edge_map
        for key in keys_remove {
            let _ = self.edge_map.remove(&key);
        }

        // Do not modify bitmap nor sequence, but remember by setting the modified-bit
        self.is_modified = true;

        // Remove from data_table_nodes
        let old_node: (L, N) = self.data_table_nodes.swap_remove(idx);

        Ok(old_node)
    }

    /// Adds a new edge to the WaveModel. May panic when either of the connected nodes does not
    /// exist inside of the WaveModel. Returns the index of the newly added edge.
    /// Refer to the `add_edge` function if additional checks should be performed.
    /// Note that this function does not return a result, but rather the index of the newly created
    /// edge directly.
    ///
    /// # Panics
    ///
    /// Will panic when supplied with at least one non-existing node.
    /// If additional checks should be performed, refer to the `add_edge` function.
    pub fn add_edge_unchecked(
        &mut self,
        label: L,
        weight: E,
        node_from: L,
        node_to: L,
        is_directed: bool,
    ) -> usize {
        // Update data_table
        self.data_table_edges.push((label, weight));
        let idx_new_edge = self.data_table_edges.len() - 1;

        // Update edge_map
        match self.edge_map.get_mut(&(node_from.clone(), node_to.clone())) {
            Some(idxs) => idxs.push(idx_new_edge),
            None => {
                self.edge_map
                    .insert((node_from.clone(), node_to.clone()), vec![idx_new_edge]);
            }
        }
        // Also add `returning` edge if model is undirected
        if !is_directed {
            match self.edge_map.get_mut(&(node_to.clone(), node_from.clone())) {
                Some(idxs) => idxs.push(idx_new_edge),
                None => {
                    self.edge_map
                        .insert((node_to.clone(), node_from.clone()), vec![idx_new_edge]);
                }
            }
        }

        // Do not modify bitmap nor sequence, but set modified bit
        self.is_modified = true;

        return idx_new_edge;
    }

    /// Adds a new edge to the WaveModel with checking beforehand if both supplied nodes actually
    /// exist inside of the current state of the WaveModel. Returns a result set potentially
    /// containing the index of the newly created edge. In case of an undirected model, only the
    /// latter index will be returned (= index of the `returning` edge; the other index can be
    /// retrieved by subtracting one).
    /// If the existence of both nodes can be assured, the `add_edge_unchecked` function can be
    /// used to save some performance going into those checks.
    pub fn add_edge(
        &mut self,
        label: L,
        weight: E,
        node_from: L,
        node_to: L,
    ) -> Result<usize, WaveModelError> {
        match self
            .data_table_nodes
            .iter()
            .filter(|(l, _)| *l == node_from || *l == node_to)
            .count()
        {
            2 => Ok(self.add_edge_unchecked(
                label.clone(),
                weight,
                node_from.clone(),
                node_to.clone(),
                self.is_directed,
            )),

            _ => Err(WaveModelError::NodeDoesNotExist),
        }
    }

    /// Allows updating the label and/or weight of an edge. Note that the tuple containing the
    /// label and weight allows for both to be `None`. In the case of either being `None`, no
    /// operationg is performed on the particular data referred to the edge.
    /// Only ever returns an error if the passed index is out of bounds. In every other case, even
    /// when no data is changed, Ok(idx) is returned.
    pub fn update_edge(
        &mut self,
        idx: usize,
        (new_label, new_weight): (Option<L>, Option<E>),
    ) -> Result<usize, WaveModelError> {
        if let Some((label, weight)) = self.data_table_edges.get_mut(idx) {
            if let Some(new_l) = new_label {
                *label = new_l;
            }
            if let Some(new_w) = new_weight {
                *weight = new_w;
            }
            Ok(idx)
        } else {
            Err(WaveModelError::EdgeDoesNotExist)
        }
    }

    /// Removes an edge from the WaveModel. Returns an error if the passed index is out of bounds.
    /// Otherwise returns the removed edge as (label_edge, weight_edge, label_node_from,
    /// label_node_to)-tupel.
    pub fn remove_edge(&mut self, idx: usize) -> Result<(L, E, L, L), WaveModelError> {
        if let None = self.data_table_edges.get(idx) {
            return Err(WaveModelError::EdgeDoesNotExist);
        }

        // Remember nodes, that the edge was connected to
        // This option can be safely unwrapped after the edge_map was worked on,
        // because the index is already guaranteed to exist and therefore the edge
        // has to have two nodes defining it resulting in an appearence in the edge_map
        let mut nodes: Option<(L, L)> = None;

        // Remove edge from edge_map
        let edge_map_iter = self.edge_map.iter_mut();
        for (key, vals) in edge_map_iter {
            if vals.contains(&idx) {
                // Get index position inside of vals vector
                // Can safely unwrap, because contains checks for existence,
                // therefore position will never return None
                let vals_idx = vals.iter().position(|n| *n == idx).unwrap();
                vals.remove(vals_idx);

                // Edge index can only be part of a single edge-connection
                nodes = Some(key.clone());

                break;
            }
        }

        if self.edge_map.get(&nodes.clone().unwrap()).unwrap().len() == 0 {
            self.edge_map.remove(&nodes.clone().unwrap());
        }

        // Do not modify bitmap nor sequence, but set modified bit
        self.is_modified = true;

        // Remove edge from data_table and swap in last edge
        let old_edge: (L, E) = self.data_table_edges.swap_remove(idx);

        // Finally return the just removed edge
        Ok((
            old_edge.0,
            old_edge.1,
            nodes.clone().unwrap().0,
            nodes.clone().unwrap().1,
        ))
    }

    /// Tries to find the corresponding index to the supplied node-label. Returns `None` if the
    /// label cannot be found.
    /// Searches in linear time.
    pub fn node_index(&self, label: &L) -> Option<usize> {
        let mut iter = self.data_table_nodes.iter();
        let mut i = 0;

        while let Some((label_found, _)) = iter.next() {
            if *label_found == *label {
                return Some(i);
            }
            i += 1;
        }
        return None;
    }

    /// Returns the corresponding label to the supplied node-index. Returns `None` if the index is
    /// out of bounds.
    pub fn node_label(&self, idx: &usize) -> Option<&L> {
        match self.data_table_nodes.get(*idx) {
            Some((l, _)) => Some(l),
            None => None,
        }
    }

    /// Tries to find the corresponding index to the supplied edge-label. Returns `None` if the
    /// label cannot be found.
    /// Searches in linear time.
    pub fn edge_index(&self, label: &L) -> Option<usize> {
        let mut iter = self.data_table_edges.iter();
        let mut i = 0;

        while let Some((label_found, _)) = iter.next() {
            if *label_found == *label {
                return Some(i);
            }
            i += 1;
        }
        return None;
    }

    /// Returns the corresponding label to the supplied edge-index. Returns `None` if the index is
    /// out of bounds.
    pub fn edge_label(&self, idx: &usize) -> Option<&L> {
        match self.data_table_edges.get(*idx) {
            Some((l, _)) => Some(l),
            None => None,
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
            is_modified: false,
        };

        Ok(wavemodel)
    }
}

#[cfg(test)]
mod test {
    use super::QWT;
    use crate::wavegraph::WaveModel;
    use std::collections::HashMap;
    use sucds::bit_vectors::{BitVector, Rank9Sel};

    fn create_empty_directed_test_model() -> WaveModel<String, usize, usize> {
        let sequence = vec![];
        let bitmap = Rank9Sel::new(BitVector::new());
        let edge_map = HashMap::<(String, String), Vec<usize>>::new();
        let data_table_nodes = vec![];
        let data_table_edges = vec![];
        let wavelet_matrix = QWT::QWT256(qwt::QWT256::from(vec![]));

        WaveModel {
            wavelet_matrix,
            sequence,
            bitmap,
            edge_map,
            data_table_nodes,
            data_table_edges,
            is_directed: true,
            is_modified: false,
        }
    }

    fn create_empty_undirected_test_model() -> WaveModel<String, usize, usize> {
        let sequence = vec![];
        let bitmap = Rank9Sel::new(BitVector::new());
        let edge_map = HashMap::<(String, String), Vec<usize>>::new();
        let data_table_nodes = vec![];
        let data_table_edges = vec![];
        let wavelet_matrix = QWT::QWT256(qwt::QWT256::from(vec![]));

        WaveModel {
            wavelet_matrix,
            sequence,
            bitmap,
            edge_map,
            data_table_nodes,
            data_table_edges,
            is_directed: false,
            is_modified: false,
        }
    }

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
            is_modified: false,
        }
    }

    fn create_undirected_test_model() -> WaveModel<String, usize, usize> {
        let sequence = vec![
            "v2".to_string(),
            "v3".to_string(),
            "v1".to_string(),
            "v3".to_string(),
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
            is_modified: false,
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

    #[test]
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
    fn check_add_node_directed() {
        let mut model = create_empty_directed_test_model();
        model.add_node("v1".to_string(), 42 as usize);

        let edge_map_expected = HashMap::<(String, String), Vec<usize>>::new();
        let data_table_nodes_expected = vec![("v1".to_string(), 42 as usize)];
        let data_table_edges_expected = vec![];
        let is_directed_expected = true;

        check_members_equal_to(
            &model,
            &(model.sequence),
            &(model.bitmap),
            &edge_map_expected,
            &data_table_nodes_expected,
            &data_table_edges_expected,
            &is_directed_expected,
        );
    }

    #[test]
    fn check_add_node_undirected() {
        let mut model = create_empty_undirected_test_model();
        model.add_node("v1".to_string(), 42 as usize);

        let edge_map_expected = HashMap::<(String, String), Vec<usize>>::new();
        let data_table_nodes_expected = vec![("v1".to_string(), 42 as usize)];
        let data_table_edges_expected = vec![];
        let is_directed_expected = false;

        check_members_equal_to(
            &model,
            &(model.sequence),
            &(model.bitmap),
            &edge_map_expected,
            &data_table_nodes_expected,
            &data_table_edges_expected,
            &is_directed_expected,
        );
    }

    #[test]
    fn check_remove_node_directed() {
        let mut model = create_directed_test_model();
        if let Ok(_) = model.remove_node(2 as usize) {
            let mut edge_map_expected = HashMap::<(String, String), Vec<usize>>::new();
            edge_map_expected.insert(("v1".to_string(), "v2".to_string()), vec![0 as usize]);
            edge_map_expected.insert(("v2".to_string(), "v1".to_string()), vec![1 as usize]);

            let data_table_nodes_expected = vec![("v1".to_string(), 0), ("v2".to_string(), 1)];
            let data_table_edges_expected = vec![("e1".to_string(), 0), ("e3".to_string(), 1)];

            let is_directed_expected = true;

            check_members_equal_to(
                &model,
                &(model.sequence),
                &(model.bitmap),
                &edge_map_expected,
                &data_table_nodes_expected,
                &data_table_edges_expected,
                &is_directed_expected,
            );
        } else {
            assert!(false, "Call to `remove_node` function failed!");
        }
    }

    #[test]
    fn check_remove_node_undirected() {
        let mut model = create_undirected_test_model();
        if let Ok(_) = model.remove_node(1 as usize) {
            let mut edge_map_expected = HashMap::<(String, String), Vec<usize>>::new();
            edge_map_expected.insert(("v1".to_string(), "v2".to_string()), vec![0 as usize]);
            edge_map_expected.insert(("v1".to_string(), "v3".to_string()), vec![1 as usize]);
            edge_map_expected.insert(("v3".to_string(), "v1".to_string()), vec![2 as usize]);
            edge_map_expected.insert(("v3".to_string(), "v2".to_string()), vec![3 as usize]);

            let data_table_nodes_expected = vec![("v1".to_string(), 0), ("v3".to_string(), 2)];
            let data_table_edges_expected = vec![
                ("e1".to_string(), 0),
                ("e2".to_string(), 1),
                ("e6".to_string(), 2),
                ("e5".to_string(), 3),
            ];

            let is_directed_expected = false;

            check_members_equal_to(
                &model,
                &(model.sequence),
                &(model.bitmap),
                &edge_map_expected,
                &data_table_nodes_expected,
                &data_table_edges_expected,
                &is_directed_expected,
            );
        } else {
            assert!(false, "Call to `remove_node` function failed!");
        }
    }

    #[test]
    fn check_add_edge_directed() {
        let mut model = create_directed_test_model();
        if let Ok(_) = model.add_edge(
            "e99".to_string(),
            42 as usize,
            "v1".to_string(),
            "v2".to_string(),
        ) {
            let mut edge_map_expected = HashMap::<(String, String), Vec<usize>>::new();
            edge_map_expected.insert(
                ("v1".to_string(), "v2".to_string()),
                vec![0 as usize, 5 as usize],
            );
            edge_map_expected.insert(("v1".to_string(), "v3".to_string()), vec![1 as usize]);
            edge_map_expected.insert(("v2".to_string(), "v1".to_string()), vec![2 as usize]);
            edge_map_expected.insert(("v3".to_string(), "v1".to_string()), vec![3 as usize]);
            edge_map_expected.insert(("v3".to_string(), "v2".to_string()), vec![4 as usize]);

            let data_table_nodes_expected = vec![
                ("v1".to_string(), 0),
                ("v2".to_string(), 1),
                ("v3".to_string(), 2),
            ];

            let data_table_edges_expected = vec![
                ("e1".to_string(), 0),
                ("e2".to_string(), 1),
                ("e3".to_string(), 2),
                ("e4".to_string(), 3),
                ("e5".to_string(), 4),
                ("e99".to_string(), 42),
            ];

            let is_directed_expected = true;

            check_members_equal_to(
                &model,
                &(model.sequence),
                &(model.bitmap),
                &edge_map_expected,
                &data_table_nodes_expected,
                &data_table_edges_expected,
                &is_directed_expected,
            );
        } else {
            assert!(false, "Call to `add_edge` function failed!");
        }
    }

    #[test]
    fn check_add_edge_undirected() {
        let mut model = create_undirected_test_model();

        if let Ok(_) = model.add_edge(
            "e99".to_string(),
            42 as usize,
            "v1".to_string(),
            "v2".to_string(),
        ) {
            let mut edge_map_expected = HashMap::<(String, String), Vec<usize>>::new();
            edge_map_expected.insert(
                ("v1".to_string(), "v2".to_string()),
                vec![0 as usize, 6 as usize],
            );
            edge_map_expected.insert(("v1".to_string(), "v3".to_string()), vec![1 as usize]);
            edge_map_expected.insert(
                ("v2".to_string(), "v1".to_string()),
                vec![2 as usize, 6 as usize],
            );
            edge_map_expected.insert(("v2".to_string(), "v3".to_string()), vec![3 as usize]);
            edge_map_expected.insert(("v3".to_string(), "v1".to_string()), vec![4 as usize]);
            edge_map_expected.insert(("v3".to_string(), "v2".to_string()), vec![5 as usize]);

            let data_table_nodes_expected = vec![
                ("v1".to_string(), 0),
                ("v2".to_string(), 1),
                ("v3".to_string(), 2),
            ];

            let data_table_edges_expected = vec![
                ("e1".to_string(), 0),
                ("e2".to_string(), 1),
                ("e3".to_string(), 2),
                ("e4".to_string(), 3),
                ("e5".to_string(), 4),
                ("e6".to_string(), 5),
                ("e99".to_string(), 42),
            ];

            let is_directed_expected = false;

            check_members_equal_to(
                &model,
                &(model.sequence),
                &(model.bitmap),
                &edge_map_expected,
                &data_table_nodes_expected,
                &data_table_edges_expected,
                &is_directed_expected,
            );
        } else {
            assert!(false, "Call to `add_edge` function failed!");
        }
    }

    #[test]
    fn check_remove_edge_directed() {
        let mut model = create_directed_test_model();
        if let Ok(_) = model.remove_edge(3 as usize) {
            let mut edge_map_expected = HashMap::<(String, String), Vec<usize>>::new();
            edge_map_expected.insert(("v1".to_string(), "v2".to_string()), vec![0 as usize]);
            edge_map_expected.insert(("v1".to_string(), "v3".to_string()), vec![1 as usize]);
            edge_map_expected.insert(("v2".to_string(), "v1".to_string()), vec![2 as usize]);
            edge_map_expected.insert(("v3".to_string(), "v2".to_string()), vec![4 as usize]);

            let data_table_nodes_expected = vec![
                ("v1".to_string(), 0),
                ("v2".to_string(), 1),
                ("v3".to_string(), 2),
            ];

            let data_table_edges_expected = vec![
                ("e1".to_string(), 0),
                ("e2".to_string(), 1),
                ("e3".to_string(), 2),
                ("e5".to_string(), 4),
            ];

            let is_directed_expected = true;

            check_members_equal_to(
                &model,
                &(model.sequence),
                &(model.bitmap),
                &edge_map_expected,
                &data_table_nodes_expected,
                &data_table_edges_expected,
                &is_directed_expected,
            );
        } else {
            assert!(false, "Call to `remove_edge` function failed!");
        }
    }

    #[test]
    fn check_remove_edge_undirected() {
        let mut model = create_undirected_test_model();
        if let Ok(_) = model.remove_edge(2 as usize) {
            let mut edge_map_expected = HashMap::<(String, String), Vec<usize>>::new();
            edge_map_expected.insert(("v1".to_string(), "v2".to_string()), vec![0 as usize]);
            edge_map_expected.insert(("v1".to_string(), "v3".to_string()), vec![1 as usize]);
            edge_map_expected.insert(("v2".to_string(), "v3".to_string()), vec![3 as usize]);
            edge_map_expected.insert(("v3".to_string(), "v1".to_string()), vec![4 as usize]);
            edge_map_expected.insert(("v3".to_string(), "v2".to_string()), vec![5 as usize]);

            let data_table_nodes_expected = vec![
                ("v1".to_string(), 0),
                ("v2".to_string(), 1),
                ("v3".to_string(), 2),
            ];

            let data_table_edges_expected = vec![
                ("e1".to_string(), 0),
                ("e2".to_string(), 1),
                ("e6".to_string(), 5),
                ("e4".to_string(), 3),
                ("e5".to_string(), 4),
            ];

            let is_directed_expected = false;

            check_members_equal_to(
                &model,
                &(model.sequence),
                &(model.bitmap),
                &edge_map_expected,
                &data_table_nodes_expected,
                &data_table_edges_expected,
                &is_directed_expected,
            );
        } else {
            assert!(false, "Call to `remove_edge` function failed!");
        }
    }

    fn check_members_equal_to<
        L: std::fmt::Debug + std::cmp::PartialEq + std::hash::Hash + std::cmp::Ord,
        N: std::fmt::Debug + std::cmp::PartialEq,
        E: std::fmt::Debug + std::cmp::PartialEq,
    >(
        model: &WaveModel<L, N, E>,
        sequence_expected: &Vec<L>,
        bitmap_expected: &Rank9Sel,
        edge_map_expected: &HashMap<(L, L), Vec<usize>>,
        data_table_nodes_expected: &Vec<(L, N)>,
        data_table_edges_expected: &Vec<(L, E)>,
        is_directed_expected: &bool,
    ) {
        let sequence_found = &model.sequence;
        assert!(
            *sequence_found == *sequence_expected,
            "Sequence is not as expected!\nExpected: {0:?}\nFound: {1:?}",
            sequence_expected,
            sequence_found
        );

        let bitmap_found = &model.bitmap;
        assert!(
            *bitmap_found == *bitmap_expected,
            "Bitmap is not as expected!\nExpected: {0:?}\nFound: {1:?}",
            bitmap_expected,
            bitmap_found
        );

        let edge_map_found = &model.edge_map;
        assert!(
            *edge_map_found == *edge_map_expected,
            "edge_map is not as expected!\nExpected: {0:?}\nFound: {1:?}",
            edge_map_expected,
            edge_map_found
        );

        let data_table_nodes_found = &model.data_table_nodes;
        assert!(
            *data_table_nodes_found == *data_table_nodes_expected,
            "data_table_nodes is not as expected!\nExpected: {0:?}\nFound: {1:?}",
            data_table_nodes_expected,
            data_table_nodes_found
        );

        let data_table_edges_found = &model.data_table_edges;
        assert!(
            *data_table_edges_found == *data_table_edges_expected,
            "data_table_edges is not as expected!\nExpected: {0:?}\nFound: {1:?}",
            data_table_edges_expected,
            data_table_edges_found
        );

        let is_directed_found = &model.is_directed;
        assert!(
            *is_directed_found == *is_directed_expected,
            "WaveModel direction has unexpectedly changed!",
        );
    }
}
