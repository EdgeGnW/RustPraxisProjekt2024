#![allow(dead_code)]

use petgraph::{
    csr::DefaultIx,
    graph::{DiGraph, EdgeIndex, NodeIndex, UnGraph},
    Graph,
};

/// Allows for the selection of either a 256-bit-alphabet-length wavelet-matrix or 
/// the 512-bit variant.
pub enum QWT<L> {
    QWT256(qwt::QWT256<L>),
    QWT512(qwt::QWT512<L>),
}

//L - Label | node/edge addressing
//N - Node data type
//E - Edge data type
//Ty - EdgeType | Directed or Undirected
//Ix - IndexType | node/edge indexing. u8/u16/u32/usize. https://docs.rs/petgraph/latest/petgraph/graph/trait.IndexType.html#foreign-impls

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
    Ty: petgraph::EdgeType,
    Ix: petgraph::adj::IndexType,
{
    graph: Graph<L, L, Ty, Ix>,
    data_table_nodes: Vec<(L, N)>,
    data_table_edges: Vec<(L, E)>,
}

impl<L, N, E, Ix> GraphModel<L, N, E, petgraph::Directed, Ix>
where
    Ix: petgraph::adj::IndexType,
{
    /// Constructs a new directed GraphModel.
    pub fn new_directed() -> Self {
        GraphModel {
            graph: DiGraph::<L, L, Ix>::default(),
            data_table_nodes: Vec::<(L, N)>::new(),
            data_table_edges: Vec::<(L, E)>::new(),
        }
    }

    /// Generates an adjacency list from the current state of the underlying, directed graph.
    pub fn to_adjacency_list(&self) -> Vec<Vec<&L>> {
        let mut nodes_iter = self.graph.node_indices();
        let mut adjacency_list = Vec::<Vec<&L>>::new();

        while let Some(idx) = nodes_iter.next() {
            adjacency_list.push(match self
                .graph
                .neighbors_directed(idx, petgraph::Direction::Outgoing)
                .map(|x| self.graph.node_weight(x))
                .collect::<Option<Vec<&L>>>() {
                Some(mut a) => { a.reverse(); a },
                _ => {Vec::<&L>::new()}
            });
        }

        adjacency_list
    }
}

impl<L, N, E> GraphModel<L, N, E, petgraph::Undirected> {
    /// Constructs a new directed GraphModel.
    pub fn new_undirected() -> Self {
        GraphModel {
            graph: UnGraph::<L, L>::default(),
            data_table_nodes: Vec::<(L, N)>::new(),
            data_table_edges: Vec::<(L, E)>::new(),
        }
    }

    /// Generates an adjacency list from the current state of the underlying, undirected graph.
    pub fn to_adjacency_list(&self) -> Vec<Vec<&L>> {
        let mut nodes_iter = self.graph.node_indices();
        let mut adjacency_list = Vec::<Vec<&L>>::new();

        while let Some(idx) = nodes_iter.next() {
            adjacency_list.push(match self
                .graph
                .neighbors(idx)
                .map(|x| self.graph.node_weight(x))
                .collect::<Option<Vec<&L>>>() {
                Some(mut a) => { a.reverse(); a },
                _ => {Vec::<&L>::new()}
            });
        }

        adjacency_list
    }
}

impl<L, N, E, Ty, Ix> GraphModel<L, N, E, Ty, Ix>
where
    Ty: petgraph::EdgeType,
    Ix: petgraph::adj::IndexType,
    L: Clone,
{
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

    /// Returns whether the graph is directed or not.
    pub fn is_directed(&self) -> bool {
        self.graph.is_directed()
    }

    //TODO: Implement the other functionalities like from_edges

    pub fn into_wave_model(&self) -> (QWT<L>, Vec<N>, Vec<E>, bool) {
        //Returns a QWT either 256 or 512 with both the nodes and edges data_tables for data lookup

        //A node consist of two things a label and the index for the data.
        //The same is true for an edge.
        //Question how do we encode this into a graph?
        todo!()
    }
}

// GraphModel, where type L also implements PartialEquality
impl<L, N, E, Ty, Ix> GraphModel<L, N, E, Ty, Ix>
where
    Ty: petgraph::EdgeType,
    Ix: petgraph::adj::IndexType,
    L: Clone + PartialEq, 
{
    /// Returns the weight of a node.
    /// Returns None if the index cannot be found.
    pub fn node_weight(&self, idx: NodeIndex<Ix>) -> Option<&N> {
        if let Some(label) = self.node_label(idx) {
            while let Some((l, w)) = self.data_table_nodes.iter().next() {
                if *label == *l {
                    return Some(w);
                }
            }
        }
        return None;
    }

    /// Returns the weight of an edge.
    /// Return None if the index cannot be found.
    pub fn edge_weight(&self, idx: EdgeIndex<Ix>) -> Option<&E> {
        if let Some(label) = self.edge_label(idx) {
            while let Some((l, w)) = self.data_table_edges.iter().next() {
                if *label == *l {
                    return Some(w);
                }
            }
        }
        return None;
    }

    /// Changes the label at the node with specificied index. Returns an option containing the old
    /// label from before the change.
    /// Returns None if the node could not be found via the given index.
    pub fn update_node_label(&mut self, idx: NodeIndex<Ix>, new_label: L) -> Option<L> {
        let mut label_old: Option<L> = None;
        match self.graph.node_weight_mut(idx) {
            Some(w) => {
                label_old = Some(w.clone());
                *w = new_label.clone();
            },
            None => { return None; },
        }

        // Update on separate label-weight-vector
        match label_old {
            Some(ref old) => {
                if let Some((current, _)) = self.data_table_nodes.iter_mut().find(|(l, _)| l == old) {
                    *current = new_label;
                }        
            },
            None => {},
        }
        label_old
    }

    /// Changes the label at the edge with specified index. Returns an option containing the old
    /// label from before the change.
    /// Returns None if the edge could not be found via the given index.
    pub fn update_edge_label(&mut self, idx: EdgeIndex<Ix>, new_label: L) -> Option<L> {
        let mut label_old: Option<L> = None;
        match self.graph.edge_weight_mut(idx) {
            Some(w) => {
                label_old = Some(w.clone());
                *w = new_label.clone();
            },
            None => { return None; },
        }

        // Update on separate label-weight-vector
        match label_old {
            Some(ref old) => {
                if let Some((current, _)) = self.data_table_edges.iter_mut().find(|(l, _)| l == old) {
                    *current = new_label;
                }        
            },
            None => {},
        }
        label_old
    }
}
