#![allow(dead_code)]

use petgraph::{
    csr::DefaultIx,
    graph::{DiGraph, EdgeIndex, NodeIndex, UnGraph},
    Graph,
};

pub enum QWT<L> {
    QWT256(qwt::QWT256<L>),
    QWT512(qwt::QWT512<L>),
}

//L - Label | node/edge addressing
//N - Node data type
//E - Edge data type
//Ty - EdgeType | Directed or Undirected
//Ix - IndexType | node/edge indexing. u8/u16/u32/usize. https://docs.rs/petgraph/latest/petgraph/graph/trait.IndexType.html#foreign-impls

#[derive(Clone, Debug)]
pub struct Wavegraph<L, N, E, Ty, Ix = DefaultIx>
where
    Ty: petgraph::EdgeType,
    Ix: petgraph::adj::IndexType,
{
    graph: Graph<L, L, Ty, Ix>,
    data_table_nodes: Vec<(L, N)>,
    data_table_edges: Vec<(L, E)>,
}

impl<L, N, E, Ix> Wavegraph<L, N, E, petgraph::Directed, Ix>
where
    Ix: petgraph::adj::IndexType,
{
    pub fn new_directed() -> Self {
        Wavegraph {
            graph: DiGraph::<L, L, Ix>::default(),
            data_table_nodes: Vec::<(L, N)>::new(),
            data_table_edges: Vec::<(L, E)>::new(),
        }
    }

    pub fn get_adjacency_list(&self) -> Vec<Vec<&L>> {
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

impl<L, N, E> Wavegraph<L, N, E, petgraph::Undirected> {
    pub fn new_undirected() -> Self {
        Wavegraph {
            graph: UnGraph::<L, L>::default(),
            data_table_nodes: Vec::<(L, N)>::new(),
            data_table_edges: Vec::<(L, E)>::new(),
        }
    }

    pub fn get_adjacency_list(&self) -> Vec<Vec<&L>> {
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

impl<L, N, E, Ty, Ix> Wavegraph<L, N, E, Ty, Ix>
where
    Ty: petgraph::EdgeType,
    Ix: petgraph::adj::IndexType,
    L: Clone,
{
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

    //TODO: Implement the other functionalities like from_edges

    pub fn to_qwt(&self) -> (QWT<L>, Vec<N>, Vec<E>) {
        //Returns a QWT either 256 or 512 with both the nodes and edges data_tables for data lookup

        //A node consist of two things a label and the index for the data.
        //The same is true for an edge.
        //Question how do we encode this into a graph?
        todo!()
    }
}
