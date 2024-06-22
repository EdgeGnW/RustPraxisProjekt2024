use super::wavemodel::WaveModel;
use std::{collections::HashMap, fmt::Debug, hash::Hash};

use petgraph::{
    graph::{DefaultIx, DiGraph, EdgeIndex, EdgeIndices, IndexType, NodeIndex, UnGraph},
    EdgeType, Graph,
};

use sucds::bit_vectors::{prelude::*, Rank9Sel};

//L - Label | node/edge addressing
//N - Node data type
//E - Edge data type
//Ty - EdgeType | Directed or Undirected
//Ix - IndexType | node/edge indexing. u8/u16/u32/usize. https://docs.rs/petgraph/latest/petgraph/graph/trait.IndexType.html#foreign-impls

#[derive(thiserror::Error, Debug)]
pub enum GraphModelError {
    #[error("THE CONVERSION WENT WRONG. TRIED TO CONVERT FROM {wavemodel_direction} WAVEMODEL TO {graphmodel_direction} GRAPHMODEL")]
    ConversionError {
        wavemodel_direction: String,
        graphmodel_direction: String,
    },
    #[error("NODE NOT FOUND")]
    NodeNotFound,
    #[error("EDGE NOT FOUND")]
    EdgeNotFound,
}

/// The main model for everything (pet-)graph related. Handles the underlying graph as well
/// as the its label-weight-mapping.
/// Also provides functionality for converting into a WaveModel. Can either represent a directed or
/// undirected graph.
/// Has generic support for the label (L), node weight (N) and edge weight (E). The edge type (Ty)
/// has to be of either directed or undirected type. The index type (Ix) has different unsigned
/// variants ranging from u8 to u64 and eventually to usize.
#[derive(Clone, Debug)]
pub struct GraphModel<L, N, E, Ty, Ix = DefaultIx>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    graph: Graph<L, L, Ty, Ix>,
    data_table_nodes: Vec<(L, N)>,
    data_table_edges: Vec<(L, E)>,
}

impl<L, N, E, Ix> GraphModel<L, N, E, petgraph::Directed, Ix>
where
    Ix: IndexType,
    L: Clone + Ord + Hash,
{
    pub fn new_directed() -> Self {
        GraphModel {
            graph: DiGraph::<L, L, Ix>::default(),
            data_table_nodes: Vec::<(L, N)>::new(),
            data_table_edges: Vec::<(L, E)>::new(),
        }
    }
}

impl<L, N, E, Ix> GraphModel<L, N, E, petgraph::Undirected, Ix>
where
    Ix: IndexType,
    L: Clone + Ord + Hash,
{
    pub fn new_undirected() -> Self {
        GraphModel {
            graph: UnGraph::<L, L, Ix>::default(),
            data_table_nodes: Vec::<(L, N)>::new(),
            data_table_edges: Vec::<(L, E)>::new(),
        }
    }
}

impl<L, N, E, Ty, Ix> GraphModel<L, N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
    L: Clone + Ord + Hash,
{
    pub fn to_adjacency_list(&self) -> Vec<(L, Vec<L>)> {
        let nodes = self.graph.node_indices();
        let mut adjacency_list = Vec::<(L, Vec<L>)>::new();

        for node in nodes {
            let neighbors = self.graph.neighbors(node);
            let from_node = self.graph.node_weight(node).unwrap().clone();
            let mut to_nodes = Vec::new();
            for neighbor_node in neighbors {
                let label = self.graph.node_weight(neighbor_node).unwrap().clone(); //This should never fail
                                                                                    //as we are getting the
                                                                                    //nodes out of
                                                                                    //node_indices
                match to_nodes.binary_search(&label) {
                    Err(index) => to_nodes.insert(index, label),
                    Ok(_) => (),
                }
            }
            adjacency_list.push((from_node, to_nodes))
        }

        adjacency_list
    }

    pub fn into_data_tables(self) -> (Vec<(L, N)>, Vec<(L, E)>) {
        (self.data_table_nodes, self.data_table_edges)
    }

    // Get label of some node
    pub fn node_label(&self, idx: NodeIndex<Ix>) -> Option<&L> {
        self.graph.node_weight(idx)
    }

    //Here we are adding a Node. This node will be stored inside of our internal graph. The real
    //weight will be stored inside of our data_table. To access the stored data we just need the
    //index of the node as it is the same index in our data table. Preferably all the weights
    //should be De-/Serializable.
    /// Adds a new node with its own label and weight to the graph.
    /// Returns the index pointing to the newly created node.
    pub fn add_node(&mut self, label: L, weight: N) -> NodeIndex<Ix> {
        let node = self.graph.add_node(label.clone());
        self.data_table_nodes.insert(node.index(), (label, weight));
        node
    }
    //We are adding an Edge between two Nodes. The real weight is being stored in a separate Vec.
    //To access it you need to get the index of the node. That index will be the one where the data
    //is stored.Preferably all the weights should be De-/Serializable.
    pub fn edge_label(&self, idx: EdgeIndex<Ix>) -> Option<&L> {
        self.graph.edge_weight(idx)
    }
    //We are adding an Edge between two Nodes. The real weight is being stored in a separate Vec.
    //To access it you need to get the index of the node. That index will be the one where the data
    //is stored.Preferably all the weights should be De-/Serializable.
    /// Adds a new edge with its own label and weight to the graph.
    /// Returns the index pointing to the newly created edge.
    pub fn add_edge(
        &mut self,
        a: NodeIndex<Ix>,
        b: NodeIndex<Ix>,
        label: L,
        weight: E,
    ) -> EdgeIndex<Ix> {
        let edge = self.graph.add_edge(a, b, label.clone());
        self.data_table_edges.insert(edge.index(), (label, weight));
        edge
    }

    /// Removes a node from the graph. If the node was given a label, this function returns the
    /// label paired with its weight after removing both from the structure.
    pub fn remove_node(&mut self, a: NodeIndex<Ix>) -> Option<(L, N)> {
        //Remove node removes the node if it exists and it replaces it with the last node that was
        //added. The indices are also shifted accordingly.
        let _label = match self.graph.remove_node(a) {
            Some(label) => label,
            None => return None,
        };

        //This could actually panic! swap_remove is actually okay here because logically it does
        //the same thing as remove_node namely removing the element at that index and replacing it
        //with the last element of the data_table
        let label_weight = self.data_table_nodes.swap_remove(a.index());
        Some(label_weight)
    }

    /// Removes an edge from the graph. If the edge was given a label, this function returns the
    /// label paired with its weight after removing both from the structure.
    pub fn remove_edge(&mut self, e: EdgeIndex<Ix>) -> Option<(L, E)> {
        //Practically the same as with the nodes.
        let _label = match self.graph.remove_edge(e) {
            Some(label) => label,
            None => return None,
        };
        //An assertion could be done with the label from the graph and the label from the data
        //table but they should not differ. To do the assertion would also require a PartialEq
        //trait bound on L
        let label_weight = self.data_table_edges.swap_remove(e.index());

        Some(label_weight)
    }

    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        GraphModel {
            graph: Graph::with_capacity(nodes, edges),
            data_table_nodes: Vec::with_capacity(nodes),
            data_table_edges: Vec::with_capacity(edges),
        }
    }

    pub fn edge_indicies(&self) -> EdgeIndices<Ix> {
        self.graph.edge_indices()
    }

    pub fn edge_endpoints(&self, e: EdgeIndex<Ix>) -> Option<(NodeIndex<Ix>, NodeIndex<Ix>)> {
        self.graph.edge_endpoints(e)
    }

    /// Returns the bitmap necessary to construct a wavelet matrix ontop of the adjacency list.
    pub fn get_bitmap(&self, adjacency_list: Vec<(L, Vec<L>)>) -> Rank9Sel {
        let mut bit_map = Vec::with_capacity(self.graph.node_count() + self.graph.edge_count());
        for (v, vs) in adjacency_list {
            bit_map.push(true);
            for _w in vs {
                bit_map.push(false);
            }
        }
        Rank9Sel::build_from_bits(bit_map, true, true, true).expect("Couldn't build Bitmap")
    }

    /// Returns whether the graph is directed or not.
    pub fn is_directed(&self) -> bool {
        self.graph.is_directed()
    }

    //TODO: Implement the other functionalities like from_edges
    fn into_wavemodel(self) -> WaveModel<L, N, E> {
        //Returns a QWT either 256 or 512 with both the nodes and edges data_tables for data lookup

        //A node consist of two things a label and the index for the data.
        //The same is true for an edge.
        //Question how do we encode this into a graph?
        // Implementation lies in WaveModel::from
        todo!()
    }

    /// Provides the option to update a label on some node.
    /// Returns a NodeNotFound error, if the index cannot be found.
    ///
    /// Searches inside of the underlying petrgraph for the node as provided by its index and
    /// gets its label reference. Breaks and returns an error if the node cannot be found.
    /// Afterwards searches inside of the corresponding label-to-weight-table with the found label
    /// and only then updates the label on both structures.
    ///
    /// # Examples
    /// ```
    /// // TODO: Give example here
    /// ```
    pub fn update_node_label(
        &mut self,
        idx: NodeIndex<Ix>,
        new_label: L,
    ) -> Result<NodeIndex<Ix>, GraphModelError> {
        // Look for label on petgraph
        if let Some(label_ref) = self.graph.node_weight_mut(idx) {
            // Find label in data table
            let mut data_table_iter = self.data_table_nodes.iter_mut();
            while let Some((label, _)) = data_table_iter.next() {
                if *label_ref == *label {
                    // Finally change the label
                    *label = new_label.clone();
                    *label_ref = new_label.clone();
                    return Ok(idx);
                }
            }
        }
        // Otherwhise return NodeNotFound error!
        return Err(GraphModelError::NodeNotFound);
    }

    /// Provides the option to update a label on some edge.
    /// Returns a EdgeNotFound error, if the index cannot be found.
    ///
    /// Searches inside of the underlying petrgraph for the edge as provided by its index and
    /// gets its label reference. Breaks and returns an error if the edge cannot be found.
    /// Afterwards searches inside of the corresponding label-to-weight-table with the found label
    /// and only then updates the label on both structures.
    ///
    /// # Examples
    /// ```
    /// // TODO: Give example here
    /// ```
    pub fn update_edge_label(
        &mut self,
        idx: EdgeIndex<Ix>,
        new_label: L,
    ) -> Result<EdgeIndex<Ix>, GraphModelError> {
        // Look for label on petgraph
        if let Some(label_ref) = self.graph.edge_weight_mut(idx) {
            // Find label in data table
            let mut data_table_iter = self.data_table_edges.iter_mut();
            while let Some((label, _)) = data_table_iter.next() {
                if *label_ref == *label {
                    // Finally change the label
                    *label = new_label.clone();
                    *label_ref = new_label.clone();
                    return Ok(idx);
                }
            }
        }
        // Otherwhise return EdgeNotFound error!
        return Err(GraphModelError::EdgeNotFound);
    }
}

impl<L, N, E, Ix> TryFrom<WaveModel<L, N, E>> for GraphModel<L, N, E, petgraph::Directed, Ix>
where
    Ix: IndexType,
    L: Clone + Ord + Hash + Debug,
    N: Debug,
    E: Debug,
{
    type Error = GraphModelError;
    fn try_from(value: WaveModel<L, N, E>) -> Result<Self, Self::Error> {
        if value.is_directed() {
            let adjacency_list = value.to_adjacency_list();
            let edge_map = value.to_edge_map();
            let (data_table_nodes, data_table_edges) = value.into_data_tables();

            let mut temp_nodes_hash_map: HashMap<&L, NodeIndex<Ix>> = HashMap::new();

            let mut graph = Graph::with_capacity(data_table_nodes.len(), data_table_edges.len());

            for (node_label, neighbor_labels) in &adjacency_list {
                let current_node_index;
                match temp_nodes_hash_map.get(&node_label) {
                    Some(node_index) => current_node_index = node_index.clone(),
                    None => {
                        temp_nodes_hash_map.insert(&node_label, graph.add_node(node_label.clone()));
                        current_node_index = temp_nodes_hash_map.get(node_label).unwrap().clone();
                    }
                }

                for neighbor_label in neighbor_labels {
                    let current_neighbor_node_index;
                    match temp_nodes_hash_map.get(&neighbor_label) {
                        Some(node_index) => current_neighbor_node_index = node_index.clone(),
                        None => {
                            temp_nodes_hash_map
                                .insert(&neighbor_label, graph.add_node(neighbor_label.clone()));
                            current_neighbor_node_index =
                                temp_nodes_hash_map.get(neighbor_label).unwrap().clone();
                        }
                    }

                    let val = edge_map[&(node_label.clone(), neighbor_label.clone())];
                    let label = data_table_edges.get(val).unwrap().0.clone();
                    graph.add_edge(current_node_index, current_neighbor_node_index, label);
                }
            }

            let graphmodel = GraphModel {
                graph,
                data_table_nodes,
                data_table_edges,
            };

            Ok(graphmodel)
        } else {
            Err(GraphModelError::ConversionError {
                wavemodel_direction: format!("UNDIRECTED"),
                graphmodel_direction: format!("DIRECTED"),
            })
        }
    }
}

impl<L, N, E, Ix> TryFrom<WaveModel<L, N, E>> for GraphModel<L, N, E, petgraph::Undirected, Ix>
where
    Ix: IndexType,
    L: Clone + Ord + Hash,
{
    type Error = GraphModelError;
    fn try_from(value: WaveModel<L, N, E>) -> Result<Self, Self::Error> {
        if !value.is_directed() {
            let adjacency_list = value.to_adjacency_list();
            let edge_map = value.to_edge_map();
            let (data_table_nodes, data_table_edges) = value.into_data_tables();

            let mut graph = Graph::with_capacity(data_table_nodes.len(), data_table_edges.len());

            //A LOT OF CLONING :(
            for (node_label, neighbor_labels) in adjacency_list {
                let node_index = graph.add_node(node_label.clone());
                for neighbor_label in neighbor_labels {
                    let neighbor_node_index = graph.add_node(neighbor_label.clone());
                    let val = edge_map[&(node_label.clone(), neighbor_label)];
                    let label = data_table_nodes.get(val).unwrap().0.clone();
                    let edge_index = graph.update_edge(node_index, neighbor_node_index, label);

                    assert!(val == edge_index.index());
                }
            }

            let graphmodel = GraphModel {
                graph,
                data_table_nodes,
                data_table_edges,
            };

            Ok(graphmodel)
        } else {
            Err(GraphModelError::ConversionError {
                wavemodel_direction: format!("DIRECTED"),
                graphmodel_direction: format!("UNDIRECTED"),
            })
        }
    }
}

#[cfg(test)]
mod test {
    use crate::wavegraph::GraphModel;
    use crate::wavegraph::WaveModel;
    use std::collections::HashMap;
    use sucds::bit_vectors::BitVector;

    #[test]
    fn check_adjacency_list_directed() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let v1 = graph.add_node("v1", 1.0);
        let v3 = graph.add_node("v3", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v4 = graph.add_node("v4", 2.0);

        // Edges:
        // v1 -> v2
        let e1 = graph.add_edge(v1, v2, "e1", 1.0);
        // v1 -> v3
        let e2 = graph.add_edge(v1, v3, "e2", 1.0);
        // v3 -> v1
        let e3 = graph.add_edge(v3, v1, "e3", 1.0);
        // v3 -> v2
        let e4 = graph.add_edge(v3, v2, "e4", 1.0);
        // v3 -> v4
        let e5 = graph.add_edge(v3, v4, "e5", 1.0);
        // v4 -> v1
        let e6 = graph.add_edge(v4, v1, "e6", 1.0);
        // v4 -> v2
        let e7 = graph.add_edge(v4, v2, "e7", 1.0);

        let adjacency_list = graph.to_adjacency_list();

        assert!(
            adjacency_list.len() == 4,
            "Adjacency list has the wrong length!"
        );
        assert!(
            adjacency_list
                == vec![
                    ("v1", vec!["v2", "v3"]),
                    ("v3", vec!["v1", "v2", "v4"]),
                    ("v2", vec![]),
                    ("v4", vec!["v1", "v2"]),
                ]
        );
    }

    #[test]
    fn check_adjacency_list_undirected() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Undirected> =
            GraphModel::new_undirected();
        let v1 = graph.add_node("v1", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v3 = graph.add_node("v3", 1.0);

        // Edges:
        // v1 - v2
        let e1 = graph.add_edge(v1, v2, "e1", 1.0);
        // v1 - v3
        let e2 = graph.add_edge(v1, v3, "e2", 1.0);
        let e3 = graph.add_edge(v2, v3, "e3", 1.0);
        // v3 - v2
        let e4 = graph.add_edge(v3, v2, "e4", 1.0);

        let adjacency_list = graph.to_adjacency_list();

        assert!(
            adjacency_list.len() == 3,
            "Adjacency list has the wrong length!"
        );

        assert!(
            adjacency_list
                == vec![
                    ("v1", vec!["v2", "v3"]),
                    ("v2", vec!["v1", "v3"]),
                    ("v3", vec!["v1", "v2"]),
                ]
        );
    }

    #[test]
    fn check_transformation_from_graphmodel_directed() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let v1 = graph.add_node("v1", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v3 = graph.add_node("v3", 1.0);

        // Edges:
        // v1 - v2
        let e1 = graph.add_edge(v1, v2, "e1", 1.0);
        // v1 - v3
        let e2 = graph.add_edge(v1, v3, "e2", 1.0);
        // v2 - v3
        let e3 = graph.add_edge(v2, v3, "e3", 1.0);
        // v3 - v2
        let e4 = graph.add_edge(v3, v2, "e4", 1.0);

        match WaveModel::try_from(graph) {
            Ok(wavemodel) => {
                // sequence
                let sequence_exp = vec!["v2", "v3", "v3", "v2"];
                let sequence_found = wavemodel.sequence().clone();
                assert!(sequence_exp == sequence_found);

                // bitmap
                let bitmap_exp =
                    BitVector::from_bits([true, false, false, true, false, true, false]);
                let bitmap_found = wavemodel.bitmap().clone();
                assert!(bitmap_exp == bitmap_found);

                // is_directed
                assert!(wavemodel.is_directed());

                // edge_map
                let mut edge_map_expected = HashMap::new();
                edge_map_expected.insert(("v1", "v2"), 0); // e1
                edge_map_expected.insert(("v1", "v3"), 1); // e2
                edge_map_expected.insert(("v2", "v3"), 2); // e3
                edge_map_expected.insert(("v3", "v2"), 3); // e4
                let edge_map_found = wavemodel.to_edge_map();
                assert!(edge_map_found == edge_map_expected);
            }
            Err(e) => {
                assert!(false);
            }
        }
    }

    #[test]
    fn check_transformation_from_graphmodel_undirected() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Undirected> =
            GraphModel::new_undirected();
        let v1 = graph.add_node("v1", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v3 = graph.add_node("v3", 1.0);

        // Edges:
        // v1 - v2
        let e1 = graph.add_edge(v1, v2, "e1", 1.0);
        // v1 - v3
        let e2 = graph.add_edge(v1, v3, "e2", 1.0);
        // v2 - v3
        let e3 = graph.add_edge(v2, v3, "e3", 1.0);
        // v3 - v2
        let e4 = graph.add_edge(v3, v2, "e4", 1.0);

        match WaveModel::try_from(graph) {
            Ok(wavemodel) => {
                // sequence
                let sequence_exp = vec!["v2", "v3", "v1", "v3", "v1", "v2"];
                let sequence_found = wavemodel.sequence().clone();
                assert!(sequence_exp == sequence_found);

                // bitmap
                let bitmap_exp = BitVector::from_bits([
                    true, false, false, true, false, false, true, false, false,
                ]);
                let bitmap_found = wavemodel.bitmap().clone();
                assert!(bitmap_exp == bitmap_found);

                // is_directed
                assert!(!wavemodel.is_directed());

                // edge_map
                let mut edge_map_expected = HashMap::new();
                edge_map_expected.insert(("v1", "v2"), 0); // e1
                edge_map_expected.insert(("v1", "v3"), 1); // e2
                edge_map_expected.insert(("v2", "v3"), 2); // e3
                edge_map_expected.insert(("v3", "v2"), 3); // e4
                let edge_map_found = wavemodel.to_edge_map();
                assert!(edge_map_found == edge_map_expected);
            }
            Err(e) => {
                assert!(false);
            }
        }
    }

    #[test]
    fn check_transformation_from_wavemodel_directed() {
        #![allow(unused_variables)]
        let mut graph: GraphModel<&str, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let v1 = graph.add_node("v1", 1.0);
        let v2 = graph.add_node("v2", 1.5);
        let v3 = graph.add_node("v3", 1.0);

        // Edges:
        // v1 - v2
        let e1 = graph.add_edge(v1, v2, "e1", 1.0);
        // v1 - v3
        let e2 = graph.add_edge(v1, v3, "e2", 1.0);
        // v2 - v3
        let e3 = graph.add_edge(v2, v3, "e3", 1.0);
        // v3 - v2
        let e4 = graph.add_edge(v3, v2, "e4", 1.0);

        let mut graph_orig = graph.clone();

        let wavemodel: WaveModel<&str, f64, f64>;

        match WaveModel::try_from(graph) {
            Ok(w) => {
                wavemodel = w;
            }
            Err(e) => {
                wavemodel = WaveModel::new();
                assert!(false);
            }
        }

        match GraphModel::<&str, f64, f64, petgraph::prelude::Directed>::try_from(wavemodel) {
            Ok(graphmodel) => {
                let adjacency_list_expected = vec![
                    ("v1", vec!["v2", "v3"]),
                    ("v2", vec!["v3"]),
                    ("v3", vec!["v2"]),
                ];
                let adjacency_list_found = graphmodel.to_adjacency_list();

                assert!(adjacency_list_found == adjacency_list_expected);
            }
            Err(e) => {}
        }
    }
}
