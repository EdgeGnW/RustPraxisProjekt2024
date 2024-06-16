use super::wavemodel::WaveModel;
use std::hash::Hash;

use petgraph::{
    graph::{DefaultIx, DiGraph, EdgeIndex, EdgeIndices, IndexType, NodeIndex, UnGraph},
    EdgeType, Graph,
};

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
}

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

    pub fn is_directed(&self) -> bool {
        self.graph.is_directed()
    }

    //Here we are adding a Node. This node will be stored inside of our internal graph. The real
    //weight will be stored inside of our data_table. To access the stored data we just need the
    //index of the node as it is the same index in our data table. Preferably all the weights
    //should be De-/Serializable.
    pub fn add_node(&mut self, label: L, weight: N) -> NodeIndex<Ix> {
        let node = self.graph.add_node(label.clone());
        self.data_table_nodes.insert(node.index(), (label, weight));
        node
    }
    //We are adding an Edge between two Nodes. The real weight is being stored in a separate Vec.
    //To access it you need to get the index of the node. That index will be the one where the data
    //is stored.Preferably all the weights should be De-/Serializable.
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

    pub fn node_label(&self, a: NodeIndex<Ix>) -> Option<&L> {
        self.graph.node_weight(a)
    }

    pub fn edge_label(&self, e: EdgeIndex<Ix>) -> Option<&L> {
        self.graph.edge_weight(e)
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

    //TODO: Implement the other functionalities like from_edges
    fn into_wavemodel(self) -> WaveModel<L, N, E> {
        //Returns a QWT either 256 or 512 with both the nodes and edges data_tables for data lookup

        //A node consist of two things a label and the index for the data.
        //The same is true for an edge.
        //Question how do we encode this into a graph?

        // Implementation lies in WaveModel::from
        todo!()
    }
}

impl<L, N, E, Ix> TryFrom<WaveModel<L, N, E>> for GraphModel<L, N, E, petgraph::Directed, Ix>
where
    Ix: IndexType,
    L: Clone + Ord + Hash,
{
    type Error = GraphModelError;
    fn try_from(value: WaveModel<L, N, E>) -> Result<Self, Self::Error> {
        if value.is_directed() {
            let adjacency_list = value.to_adjacency_list();
            let edge_map = value.to_edge_map();
            let (data_table_nodes, data_table_edges) = value.into_data_tables();

            let mut graph = Graph::with_capacity(data_table_nodes.len(), data_table_edges.len());

            //A LOT OF CLONING :(
            for (node_label, neighbor_labels) in adjacency_list {
                let current_node_index = graph.add_node(node_label.clone());
                for neighbor_label in neighbor_labels {
                    let current_neighbor_node_index = graph.add_node(neighbor_label.clone());
                    let val = edge_map[&(node_label.clone(), neighbor_label)];
                    let label = data_table_nodes.get(val).unwrap().0.clone();
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
