mod wavegraph;

#[cfg(test)]
mod test {
    use crate::wavegraph::Wavegraph;
    use petgraph::{csr::IndexType, Undirected};
    #[test]
    fn check_something() {
        let mut graph = Wavegraph::new_undirected();
        let a = graph.add_node("Duisburg", 1.0);
        let b = graph.add_node("Gelsenkirchen", 0.5);
        let e = graph.add_edge(a, b, "A40", "34km");
    }
}
